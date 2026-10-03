"""
AstroQuantizer: Dynamic Uniform Range Quantization Engine with Straight-Through Estimator (STE)
and Fisher-Adaptive Mixed-Precision Bit Allocation. Registered as 'astro' and 'astro_quantizer'.

Scientific Novelty:
- Mixed-precision dynamic bit allocation driven by Empirical Fisher Information / Output Activation Energy.
- Optimal integer budget allocation: Channels with higher Fisher sensitivity receive higher bit-widths (up to max_bits),
  while less sensitive channels are quantized with lower bit-widths (down to min_bits), satisfying an exact average bit budget.
- Straight-Through Estimator (STE) gradients flow seamlessly across channels with heterogeneous bit allocations.
- Full robustness across LoRALinear wrappers, Conv1D layers (GPT-2), and multi-architecture models (DistilBERT, RoBERTa, GPT-2, Qwen3.5).
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
from neuralops.core.base import BaseCompressor
from neuralops.core.model_utils import replace_module
from neuralops.core.pruning_utils import get_calibration_tensors
from neuralops.core.registry import COMPRESSOR_REGISTRY
from neuralops.utils.compat import should_compress_linear

try:
    from transformers.pytorch_utils import Conv1D
except ImportError:
    Conv1D = None

logger = logging.getLogger(__name__)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Safely extract configuration attribute from None, dict, or namespace object."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class STEFakeQuantize(torch.autograd.Function):
    """
    Straight-Through Estimator (STE) fake-quantization autograd function.
    Clamps inputs to [qmin, qmax] in discrete integer space, de-quantizes back,
    and passes gradients through uninhibited across all channels.
    """

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        qmin: float | torch.Tensor,
        qmax: float | torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(x, scale, zero_point)
        # Quantize, clamp to discrete range, and dequantize
        q = torch.clamp(torch.round(x / scale) + zero_point, qmin, qmax)
        dq = (q - zero_point) * scale
        return dq

    @staticmethod
    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor | None, ...]:
        # Straight-Through Estimator passes gradient unmodified
        grad_x = grad_output.clone()
        return grad_x, None, None, None, None


def allocate_fisher_bits(
    importance: torch.Tensor,
    target_avg_bits: float = 6.0,
    min_bits: int = 2,
    max_bits: int = 8,
) -> torch.Tensor:
    """
    Allocates per-channel bit-widths b_i in [min_bits, max_bits] maximizing weighted importance
    such that sum(b_i) / N == target_avg_bits (within +/- 1/N tolerance).

    Formulation:
      max sum_i (importance_i * b_i)
      s.t. sum_i b_i <= round(N * target_avg_bits)
           min_bits <= b_i <= max_bits

    Solved via greedy knapsack: sort channels descending by importance, assign max_bits
    to top channels until budget is exhausted, then assign min_bits to remainder.
    """
    N = importance.numel()
    if N == 0:
        return torch.empty(0, dtype=torch.long, device=importance.device)

    target_budget = int(round(N * float(target_avg_bits)))
    target_budget = max(N * min_bits, min(N * max_bits, target_budget))

    assigned = torch.full((N,), min_bits, dtype=torch.long, device=importance.device)
    rem = target_budget - N * min_bits
    if rem <= 0:
        return assigned

    sorted_indices = torch.argsort(importance, descending=True)
    bit_headroom = max_bits - min_bits
    if bit_headroom <= 0:
        return assigned

    num_full = rem // bit_headroom
    remainder = rem % bit_headroom

    if num_full > 0:
        top_idx = sorted_indices[:num_full]
        assigned[top_idx] = max_bits

    if remainder > 0 and num_full < N:
        rem_idx = sorted_indices[num_full]
        assigned[rem_idx] = min_bits + remainder

    return assigned


def quantize_dequantize_tensor(
    tensor: torch.Tensor,
    bits: int = 8,
    symmetric: bool = True,
    per_channel: bool = False,
    use_ste: bool = False,
    per_channel_bits: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Uniform quantization and de-quantization supporting:
    - Fixed bitwidth (scalar bits)
    - Heterogeneous per-channel bitwidths (per_channel_bits)
    - Symmetric or asymmetric ranges
    - Straight-Through Estimator (STE) autograd graph
    """
    if tensor.numel() == 0:
        return tensor

    orig_dtype = tensor.dtype
    t = tensor.float()

    if per_channel_bits is not None and t.ndim >= 2:
        # Per-channel dynamic bit allocation
        # Shape of per_channel_bits: [out_channels]
        dim_to_reduce = tuple(range(1, t.ndim))
        b = per_channel_bits.view(-1, *([1] * (t.ndim - 1))).float().to(device=t.device)

        if symmetric:
            qmin = -torch.pow(2.0, b - 1.0)
            qmax = torch.pow(2.0, b - 1.0) - 1.0
            max_abs = torch.amax(torch.abs(t), dim=dim_to_reduce, keepdim=True)
            scale = torch.clamp(max_abs / torch.clamp(qmax, min=1.0), min=1e-8)
            zero_point = torch.zeros_like(scale)
        else:
            qmin = torch.zeros_like(b)
            qmax = torch.pow(2.0, b) - 1.0
            min_val = torch.amin(t, dim=dim_to_reduce, keepdim=True)
            max_val = torch.amax(t, dim=dim_to_reduce, keepdim=True)
            scale = torch.clamp(
                (max_val - min_val) / torch.clamp(qmax, min=1.0), min=1e-8
            )
            zero_point = torch.clamp(torch.round(-min_val / scale), qmin, qmax)
    else:
        # Standard uniform bitwidth
        if symmetric:
            qmin = float(-(2 ** (bits - 1)))
            qmax = float((2 ** (bits - 1)) - 1)
            if per_channel and t.ndim >= 2:
                dim_to_reduce = tuple(range(1, t.ndim))
                max_abs = torch.amax(torch.abs(t), dim=dim_to_reduce, keepdim=True)
                scale = torch.clamp(max_abs / max(float(qmax), 1.0), min=1e-8)
                zero_point = torch.zeros_like(scale)
            else:
                max_abs = torch.max(torch.abs(t))
                scale = torch.clamp(max_abs / max(float(qmax), 1.0), min=1e-8)
                zero_point = torch.tensor(0.0, device=t.device, dtype=t.dtype)
        else:
            qmin = 0.0
            qmax = float((2**bits) - 1)
            if per_channel and t.ndim >= 2:
                dim_to_reduce = tuple(range(1, t.ndim))
                min_val = torch.amin(t, dim=dim_to_reduce, keepdim=True)
                max_val = torch.amax(t, dim=dim_to_reduce, keepdim=True)
                scale = torch.clamp(
                    (max_val - min_val) / max(float(qmax), 1.0), min=1e-8
                )
                zero_point = torch.clamp(torch.round(-min_val / scale), qmin, qmax)
            else:
                min_val = torch.min(t)
                max_val = torch.max(t)
                scale = torch.clamp(
                    (max_val - min_val) / max(float(qmax), 1.0), min=1e-8
                )
                zero_point = torch.clamp(torch.round(-min_val / scale), qmin, qmax)

    if use_ste and t.requires_grad:
        out = STEFakeQuantize.apply(t, scale, zero_point, qmin, qmax)
    else:
        q = torch.clamp(torch.round(t / scale) + zero_point, qmin, qmax)
        out = (q - zero_point) * scale

    return out.to(dtype=orig_dtype)


class AstroLinear(nn.Module):
    """Linear layer using Astro uniform quantization with STE backpropagation support."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        bits: int = 8,
        symmetric: bool = True,
        per_channel: bool = False,
        per_channel_bits: torch.Tensor | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.nf = out_features  # Conv1D compatibility
        self.bits = bits
        self.symmetric = symmetric
        self.per_channel = per_channel

        if per_channel_bits is not None:
            self.register_buffer("per_channel_bits", per_channel_bits.clone().long())
        else:
            self.register_buffer("per_channel_bits", None)

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Module,
        bits: int = 8,
        symmetric: bool = True,
        per_channel: bool = False,
        per_channel_bits: torch.Tensor | None = None,
    ) -> AstroLinear:
        cls_name = linear.__class__.__name__.lower()
        is_conv1d = "conv1d" in cls_name or hasattr(linear, "nf")
        raw_w = linear.weight.data

        if is_conv1d:
            in_f = raw_w.shape[0]
            out_f = raw_w.shape[1]
            w_data = raw_w.t()
        else:
            out_f = raw_w.shape[0]
            in_f = raw_w.shape[1]
            w_data = raw_w

        layer = cls(
            in_features=in_f,
            out_features=out_f,
            bias=getattr(linear, "bias", None) is not None and linear.bias is not None,
            bits=bits,
            symmetric=symmetric,
            per_channel=per_channel,
            per_channel_bits=per_channel_bits,
        )
        layer.weight.data.copy_(w_data)
        if getattr(linear, "bias", None) is not None and linear.bias is not None:
            layer.bias.data.copy_(linear.bias.data)
        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_q = quantize_dequantize_tensor(
            self.weight,
            bits=self.bits,
            symmetric=self.symmetric,
            per_channel=self.per_channel or (self.per_channel_bits is not None),
            use_ste=self.training,
            per_channel_bits=self.per_channel_bits,
        )
        b_q = None
        if self.bias is not None:
            bias_bits = (
                int(torch.round(self.per_channel_bits.float().mean()).item())
                if self.per_channel_bits is not None
                else self.bits
            )
            b_q = quantize_dequantize_tensor(
                self.bias,
                bits=bias_bits,
                symmetric=self.symmetric,
                per_channel=False,
                use_ste=self.training,
            )
        return nn.functional.linear(
            x,
            w_q.to(dtype=x.dtype, device=x.device),
            b_q.to(dtype=x.dtype, device=x.device) if b_q is not None else None,
        )


@COMPRESSOR_REGISTRY.register("astro")
@COMPRESSOR_REGISTRY.register("astro_quantizer")
class AstroQuantizer(BaseCompressor):
    """
    AstroQuantizer: Uniform min-max / symmetric quantization engine with Fisher-adaptive bit allocation.

    Production hardening features:
    - Support for LoRALinear wrappers, nn.Linear, and Conv1D (GPT-2).
    - Novel Fisher-Adaptive Bit Allocation allocating mixed-precision bit-widths [min_bits, max_bits] per output channel.
    - target_attention_only parameter skipping FFN/MLP and preserving classification heads.
    - Robust config handling (_cfg_get) supporting None, dict, and namespace objects.
    - Fallback freezing: parameters in any fallback layer are frozen (requires_grad=False).
    - Full FP32 quantization calculation with NaN/Inf guards.
    - Straight-Through Estimator (STE) autograd flow.
    - Transparent metric tracking: actual_average_bits, effective_compression_rate, and theoretical_vram_saving.
    """

    def __init__(
        self,
        model: nn.Module,
        bits: int = 8,
        symmetric: bool = True,
        per_channel: bool = False,
        use_ste: bool = False,
        max_error: float = 0.20,
        fallback_to_original: bool = True,
        protect_critical: bool = True,
        target_modules: list[str] | None = None,
        target_attention_only: bool = False,
        use_fisher_adaptive_bits: bool = False,
        target_avg_bits: float = 6.0,
        min_bits: int = 2,
        max_bits: int = 8,
        fisher_beta: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="astro", **kwargs)
        cfg = kwargs.get("comp_config", kwargs.get("cfg"))

        bits = int(_cfg_get(cfg, "astro_bits", _cfg_get(cfg, "bits", bits)))
        symmetric = bool(
            _cfg_get(cfg, "astro_symmetric", _cfg_get(cfg, "symmetric", symmetric))
        )
        per_channel = bool(
            _cfg_get(
                cfg, "astro_per_channel", _cfg_get(cfg, "per_channel", per_channel)
            )
        )
        use_ste = bool(
            _cfg_get(cfg, "astro_use_ste", _cfg_get(cfg, "use_ste", use_ste))
        )
        max_error = float(
            _cfg_get(cfg, "astro_max_error", _cfg_get(cfg, "max_error", max_error))
        )
        fallback_to_original = bool(
            _cfg_get(
                cfg,
                "astro_fallback_to_original",
                _cfg_get(cfg, "fallback_to_original", fallback_to_original),
            )
        )
        protect_critical = bool(_cfg_get(cfg, "protect_critical", protect_critical))
        target_attention_only = bool(
            _cfg_get(
                cfg,
                "astro_target_attention_only",
                _cfg_get(cfg, "target_attention_only", target_attention_only),
            )
        )
        use_fisher_adaptive_bits = bool(
            _cfg_get(
                cfg,
                "astro_use_fisher_adaptive_bits",
                _cfg_get(cfg, "use_fisher_adaptive_bits", use_fisher_adaptive_bits),
            )
        )
        target_avg_bits = float(
            _cfg_get(
                cfg,
                "astro_target_avg_bits",
                _cfg_get(cfg, "target_avg_bits", target_avg_bits),
            )
        )
        min_bits = int(
            _cfg_get(cfg, "astro_min_bits", _cfg_get(cfg, "min_bits", min_bits))
        )
        max_bits = int(
            _cfg_get(cfg, "astro_max_bits", _cfg_get(cfg, "max_bits", max_bits))
        )
        fisher_beta = float(
            _cfg_get(
                cfg, "astro_fisher_beta", _cfg_get(cfg, "fisher_beta", fisher_beta)
            )
        )

        self.bits = bits
        self.symmetric = symmetric
        self.per_channel = per_channel
        self.use_ste = use_ste
        self.max_error = max_error
        self.fallback_to_original = fallback_to_original
        self.protect_critical = protect_critical
        self.target_attention_only = target_attention_only
        self.use_fisher_adaptive_bits = use_fisher_adaptive_bits
        self.target_avg_bits = target_avg_bits
        self.min_bits = min_bits
        self.max_bits = max_bits
        self.fisher_beta = fisher_beta
        self.tokenizer = kwargs.get("tokenizer", getattr(model, "tokenizer", None))
        self.calibration_loader = kwargs.get("calibration_loader")
        self.calibration_texts = kwargs.get("calibration_texts")
        device = kwargs.get("device")
        self.target_device = (
            torch.device(device)
            if device is not None
            else next(model.parameters(), torch.tensor(0.0)).device
        )
        self.comp_config = cfg
        self.target_modules = target_modules

        self.compressed_layers: list[str] = []
        self.fallback_layers: list[str] = []
        self.layer_errors: dict[str, float] = {}
        self.layer_bit_assignments: dict[str, list[int]] = {}
        self.layer_average_bits: dict[str, float] = {}

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        if isinstance(module, AstroLinear):
            return False

        # Do not target inner submodules of LoRA or wrappers directly
        if any(
            name.endswith(sfx)
            for sfx in [".base_layer", ".lora_A", ".lora_B", ".lora_dropout", ".linear"]
        ):
            return False

        target_mod = getattr(module, "base_layer", getattr(module, "linear", module))
        cls_name = module.__class__.__name__.lower()
        target_cls = target_mod.__class__.__name__.lower()

        is_linear_like = (
            isinstance(module, nn.Linear)
            or isinstance(target_mod, nn.Linear)
            or "conv1d" in cls_name
            or "conv1d" in target_cls
            or hasattr(module, "nf")
            or hasattr(target_mod, "nf")
        )
        if not is_linear_like:
            return False

        name_lower = name.lower()
        # Protect classification / output heads
        if any(
            skip in name_lower
            for skip in [
                "classifier",
                "score",
                "lm_head",
                "pooler",
                "embed",
                "logits_processor",
                "pre_classifier",
                "head",
            ]
        ):
            return False

        if self.target_attention_only:
            # Skip FFN / MLP layers
            ffn_patterns = [
                "ffn",
                "mlp",
                "lin1",
                "lin2",
                "intermediate",
                "gate_proj",
                "up_proj",
                "down_proj",
                "c_fc",
            ]
            if any(p in name_lower for p in ffn_patterns):
                return False
            # In RoBERTa, layer.X.output.dense is FFN, while attention.output.dense is attention
            if "output.dense" in name_lower and "attention" not in name_lower:
                return False
            # Match attention projections
            attn_patterns = [
                "attn",
                "attention",
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "out_lin",
                "q_lin",
                "k_lin",
                "v_lin",
                "c_attn",
                "c_proj",
                "query",
                "key",
                "value",
                "in_proj_qkv",
                "out_proj",
            ]
            if not any(p in name_lower for p in attn_patterns):
                return False
        else:
            if self.protect_critical and not should_compress_linear(self.model, name):
                return False

        if self.target_modules is not None:
            return any(t in name for t in self.target_modules)
        return True

    def _collect_fisher_info(self) -> dict[str, torch.Tensor]:
        """
        Collect empirical Fisher gradient squared or activation energy per layer.
        Returns a dict: layer_name -> 1D tensor of importance per output channel.
        """
        fisher_info: dict[str, torch.Tensor] = {}
        if not self.use_fisher_adaptive_bits:
            return fisher_info

        activations: dict[str, list[torch.Tensor]] = {}
        hooks = []

        def make_hook(layer_name: str):
            def hook(module: nn.Module, inp: tuple[torch.Tensor, ...], outp: Any):
                if inp and isinstance(inp[0], torch.Tensor):
                    x = inp[0].detach()
                    if layer_name not in activations:
                        activations[layer_name] = []
                    activations[layer_name].append(x.cpu())

            return hook

        for name, module in self.model.named_modules():
            if self._is_target_module(name, module):
                target_mod = getattr(
                    module, "base_layer", getattr(module, "linear", module)
                )
                hooks.append(target_mod.register_forward_hook(make_hook(name)))

        calib_inputs = get_calibration_tensors(
            self.model,
            tokenizer=self.tokenizer,
            calibration_loader=self.calibration_loader,
            calibration_texts=self.calibration_texts,
            device=self.target_device,
        )

        try:
            self.model.train()
            out = None
            if isinstance(calib_inputs, dict):
                out = self.model(**calib_inputs)
            elif isinstance(calib_inputs, torch.Tensor):
                out = self.model(calib_inputs)
            elif isinstance(calib_inputs, (list, tuple)):
                out = self.model(*calib_inputs)

            if out is not None:
                if hasattr(out, "logits"):
                    loss = 0.5 * torch.sum(out.logits.pow(2))
                elif isinstance(out, torch.Tensor):
                    loss = 0.5 * torch.sum(out.pow(2))
                elif (
                    isinstance(out, (tuple, list))
                    and len(out) > 0
                    and isinstance(out[0], torch.Tensor)
                ):
                    loss = 0.5 * torch.sum(out[0].pow(2))
                else:
                    loss = None

                if loss is not None:
                    self.model.zero_grad()
                    loss.backward()
                    for name, module in self.model.named_modules():
                        if self._is_target_module(name, module):
                            target_mod = getattr(
                                module, "base_layer", getattr(module, "linear", module)
                            )
                            if getattr(target_mod.weight, "grad", None) is not None:
                                g = target_mod.weight.grad.detach().float()
                                cls_name = module.__class__.__name__.lower()
                                is_conv1d = "conv1d" in cls_name or hasattr(
                                    module, "nf"
                                )
                                if is_conv1d:
                                    g = g.t()
                                per_channel_fisher = torch.sum(g.pow(2), dim=1).cpu()
                                fisher_info[name] = per_channel_fisher
                    self.model.zero_grad()
        except Exception as e:
            logger.debug("Astro Fisher calibration exception: %s", e)
        finally:
            self.model.eval()
            for h in hooks:
                h.remove()

        # Fallback to activation energy or weight L2 norm per channel for layers without gradient Fisher
        for name, module in self.model.named_modules():
            if self._is_target_module(name, module) and name not in fisher_info:
                target_mod = getattr(
                    module, "base_layer", getattr(module, "linear", module)
                )
                raw_w = target_mod.weight.data.float()
                cls_name = module.__class__.__name__.lower()
                is_conv1d = "conv1d" in cls_name or hasattr(module, "nf")
                W = raw_w.t() if is_conv1d else raw_w
                channel_norm = torch.norm(W, p=2, dim=1).cpu()
                if name in activations and activations[name]:
                    act_list = activations[name]
                    act_concat = torch.cat(
                        [a.reshape(-1, a.shape[-1]) for a in act_list], dim=0
                    ).float()
                    act_scale = torch.norm(act_concat, p=2, dim=0).mean().item()
                    channel_norm = channel_norm * max(act_scale, 1e-4)
                fisher_info[name] = channel_norm

        return fisher_info

    def compress(self) -> tuple[nn.Module, dict[str, Any]]:
        """Quantize all target linear/Conv1D layers in the model and return (compressed_model, stats)."""
        fisher_map = (
            self._collect_fisher_info() if self.use_fisher_adaptive_bits else {}
        )

        self.compressed_layers = []
        self.fallback_layers = []
        self.layer_errors = {}
        self.layer_bit_assignments = {}
        self.layer_average_bits = {}

        for name, module in list(self.model.named_modules()):
            if not self._is_target_module(name, module):
                continue

            target_mod = getattr(
                module, "base_layer", getattr(module, "linear", module)
            )
            cls_name = module.__class__.__name__.lower()
            is_conv1d = (
                "conv1d" in cls_name
                or hasattr(module, "nf")
                or hasattr(target_mod, "nf")
            )
            has_lora = (
                hasattr(module, "lora_A") and module.lora_A is not None
            ) or hasattr(module, "base_layer")

            raw_w = target_mod.weight.data
            W = raw_w.t().clone().float() if is_conv1d else raw_w.clone().float()
            out_features, in_features = W.shape

            # Determine bit-width allocation
            per_channel_bits = None
            if self.use_fisher_adaptive_bits:
                imp = fisher_map.get(name, None)
                if imp is None or imp.numel() != out_features:
                    imp = torch.norm(W, p=2, dim=1).cpu()
                imp = (imp.float() + 1e-8).pow(self.fisher_beta)
                per_channel_bits = allocate_fisher_bits(
                    imp,
                    target_avg_bits=self.target_avg_bits,
                    min_bits=self.min_bits,
                    max_bits=self.max_bits,
                )
                avg_bits = float(per_channel_bits.float().mean().item())
                self.layer_bit_assignments[name] = per_channel_bits.tolist()
                self.layer_average_bits[name] = avg_bits
            else:
                avg_bits = float(self.bits)
                self.layer_average_bits[name] = avg_bits

            # Quantize & de-quantize evaluation
            W_quant = quantize_dequantize_tensor(
                W,
                bits=self.bits,
                symmetric=self.symmetric,
                per_channel=self.per_channel or (per_channel_bits is not None),
                use_ste=False,
                per_channel_bits=per_channel_bits,
            )

            w_norm = torch.norm(W, p="fro").item()
            err = float(torch.norm(W - W_quant, p="fro").item() / max(w_norm, 1e-12))
            self.layer_errors[name] = err

            if (
                torch.isnan(W_quant).any()
                or torch.isinf(W_quant).any()
                or err > self.max_error
            ):
                logger.warning(
                    "AstroQuantizer error %.4f on %s exceeded threshold %.4f.",
                    err,
                    name,
                    self.max_error,
                )
                if self.fallback_to_original:
                    logger.info(
                        "Skipping AstroQuantizer for %s (fallback to original).", name
                    )
                    self.fallback_layers.append(name)
                    # Freeze fallback layers to ensure recovery fine-tuning does not touch them
                    for p in module.parameters():
                        p.requires_grad = False
                    continue

            # Update module: AstroLinear (STE) or In-Place weights
            if self.use_ste:
                astro_layer = AstroLinear.from_linear(
                    target_mod,
                    bits=self.bits,
                    symmetric=self.symmetric,
                    per_channel=self.per_channel or (per_channel_bits is not None),
                    per_channel_bits=per_channel_bits,
                )
                if has_lora:
                    module.base_layer = astro_layer
                elif name == "":
                    self.model = astro_layer
                else:
                    replace_module(self.model, name, astro_layer)
            else:
                if is_conv1d:
                    target_mod.weight.data.copy_(
                        W_quant.t()
                        .contiguous()
                        .to(dtype=raw_w.dtype, device=raw_w.device)
                    )
                else:
                    target_mod.weight.data.copy_(
                        W_quant.contiguous().to(dtype=raw_w.dtype, device=raw_w.device)
                    )

                if target_mod.bias is not None:
                    bias_bits = int(round(avg_bits))
                    b_quant = quantize_dequantize_tensor(
                        target_mod.bias.data.float(),
                        bits=bias_bits,
                        symmetric=self.symmetric,
                        per_channel=False,
                        use_ste=False,
                    )
                    target_mod.bias.data.copy_(
                        b_quant.to(
                            dtype=target_mod.bias.dtype, device=target_mod.bias.device
                        )
                    )

            # Preserve LoRA adapter parameters if layer was LoRALinear wrapper
            if has_lora:
                for p_name, p in module.named_parameters():
                    if any(
                        t in p_name.lower() for t in ("lora_a", "lora_b", "adapter")
                    ):
                        p.requires_grad = True

            self.compressed_layers.append(name)

        self._compressed_params = self._count_params(self.model)
        avg_err = (
            sum(self.layer_errors[k] for k in self.compressed_layers)
            / max(1, len(self.compressed_layers))
            if self.compressed_layers
            else 0.0
        )
        actual_avg_bits = (
            sum(self.layer_average_bits[k] for k in self.compressed_layers)
            / max(1, len(self.compressed_layers))
            if self.compressed_layers
            else float(self.bits)
        )
        effective_compression_rate = 16.0 / max(actual_avg_bits, 1e-3)
        theoretical_vram_saving = f"{(1.0 - actual_avg_bits / 16.0) * 100:.1f}% vs FP16"

        stats = self.get_compression_stats()
        stats.update(
            {
                "quantization_bits": self.bits,
                "actual_average_bits": actual_avg_bits,
                "symmetric": self.symmetric,
                "per_channel": self.per_channel,
                "use_ste": self.use_ste,
                "use_fisher_adaptive_bits": self.use_fisher_adaptive_bits,
                "target_avg_bits": self.target_avg_bits,
                "min_bits": self.min_bits,
                "max_bits": self.max_bits,
                "compressed_layers": self.compressed_layers,
                "fallback_layers": self.fallback_layers,
                "num_compressed": len(self.compressed_layers),
                "num_fallback": len(self.fallback_layers),
                "layer_errors": self.layer_errors,
                "layer_average_bits": self.layer_average_bits,
                "avg_reconstruction_error": avg_err,
                "max_error": self.max_error,
                "fallback_to_original": self.fallback_to_original,
                "target_attention_only": self.target_attention_only,
                "effective_compression_rate": effective_compression_rate,
                "theoretical_vram_saving": theoretical_vram_saving,
            }
        )
        self.stats = stats
        self.model.compression_stats = stats
        return self.model, stats


@COMPRESSOR_REGISTRY.register("astro_factory")
@COMPRESSOR_REGISTRY.register("astro_quantizer_factory")
def astro_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    """Astro compressor factory function."""
    bits = kwargs.pop(
        "bits", _cfg_get(comp_config, "astro_bits", _cfg_get(comp_config, "bits", 8))
    )
    symmetric = kwargs.pop(
        "symmetric",
        _cfg_get(
            comp_config, "astro_symmetric", _cfg_get(comp_config, "symmetric", True)
        ),
    )
    per_channel = kwargs.pop(
        "per_channel",
        _cfg_get(
            comp_config,
            "astro_per_channel",
            _cfg_get(comp_config, "per_channel", False),
        ),
    )
    use_ste = kwargs.pop(
        "use_ste",
        _cfg_get(comp_config, "astro_use_ste", _cfg_get(comp_config, "use_ste", False)),
    )
    max_error = kwargs.pop(
        "max_error",
        _cfg_get(
            comp_config, "astro_max_error", _cfg_get(comp_config, "max_error", 0.20)
        ),
    )
    fallback = kwargs.pop(
        "fallback_to_original",
        _cfg_get(
            comp_config,
            "astro_fallback_to_original",
            _cfg_get(comp_config, "fallback_to_original", True),
        ),
    )
    target_attention_only = kwargs.pop(
        "target_attention_only",
        _cfg_get(
            comp_config,
            "astro_target_attention_only",
            _cfg_get(comp_config, "target_attention_only", False),
        ),
    )
    use_fisher_adaptive_bits = kwargs.pop(
        "use_fisher_adaptive_bits",
        _cfg_get(
            comp_config,
            "astro_use_fisher_adaptive_bits",
            _cfg_get(comp_config, "use_fisher_adaptive_bits", False),
        ),
    )
    target_avg_bits = kwargs.pop(
        "target_avg_bits",
        _cfg_get(
            comp_config,
            "astro_target_avg_bits",
            _cfg_get(comp_config, "target_avg_bits", 6.0),
        ),
    )
    min_bits = kwargs.pop(
        "min_bits",
        _cfg_get(comp_config, "astro_min_bits", _cfg_get(comp_config, "min_bits", 2)),
    )
    max_bits = kwargs.pop(
        "max_bits",
        _cfg_get(comp_config, "astro_max_bits", _cfg_get(comp_config, "max_bits", 8)),
    )
    fisher_beta = kwargs.pop(
        "fisher_beta",
        _cfg_get(
            comp_config, "astro_fisher_beta", _cfg_get(comp_config, "fisher_beta", 0.5)
        ),
    )

    compressor = AstroQuantizer(
        model,
        bits=bits,
        symmetric=symmetric,
        per_channel=per_channel,
        use_ste=use_ste,
        max_error=max_error,
        fallback_to_original=fallback,
        target_attention_only=target_attention_only,
        use_fisher_adaptive_bits=use_fisher_adaptive_bits,
        target_avg_bits=target_avg_bits,
        min_bits=min_bits,
        max_bits=max_bits,
        fisher_beta=fisher_beta,
        comp_config=comp_config,
        **kwargs,
    )
    compressed_model, _ = compressor.compress()
    return compressed_model


__all__ = [
    "STEFakeQuantize",
    "allocate_fisher_bits",
    "quantize_dequantize_tensor",
    "AstroLinear",
    "AstroQuantizer",
    "astro_factory",
]
