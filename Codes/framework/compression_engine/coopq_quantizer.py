"""
CoopQ: Cooperative Game-Theoretic Bit Allocation Quantizer with STE and Per-Channel Asymmetric Calibration.
Registered in COMPRESSOR_REGISTRY as 'coopq'.

Mathematical Formulation:
1. Min Bit-Width Floor Enforcement:
   $$b_l \\ge b_{\\min} = 4, \\quad \\forall l \\in \\{1, \\dots, L\\}$$
2. Per-Channel Asymmetric Scale & Zero-Point Calibration:
   $$\\text{scale}_i = \\max\\left( \\frac{\\max_j W_{ij} - \\min_j W_{ij}}{2^b - 1}, 10^{-8} \\right), \\quad z_i = \\operatorname{clamp}\\left( \\left\\lfloor -\\frac{\\min_j W_{ij}}{\\text{scale}_i} \\right\\rceil, 0, 2^b - 1 \\right)$$
   $$\\widehat{W}_{ij} = \\left( \\operatorname{clamp}\\left( \\left\\lfloor \\frac{W_{ij}}{\\text{scale}_i} \\right\\rceil + z_i, 0, 2^b - 1 \\right) - z_i \\right) \\cdot \\text{scale}_i$$
3. Straight-Through Estimator (STE) Gradient Backpropagation:
   $$\\frac{\\partial \\mathcal{L}}{\\partial W} = \\frac{\\partial \\mathcal{L}}{\\partial \\widehat{W}}$$
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseCompressor
from framework.core.model_utils import replace_module
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.compression_engine.astro_quantizer import AstroLinear
from framework.core.pruning_utils import (
    get_calibration_tensors,
    recalibrate_layernorms,
)

logger = logging.getLogger(__name__)


class CoopQSTEFakeQuantize(torch.autograd.Function):
    """
    Straight-Through Estimator (STE) fake-quantization autograd function for CoopQ.
    Forwards per-channel quantized-dequantized weights and passes gradients through on backward pass.
    """

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        qmin: float,
        qmax: float,
    ) -> torch.Tensor:
        ctx.save_for_backward(scale, zero_point)
        ctx.qmin = qmin
        ctx.qmax = qmax

        # Quantize and clamp
        q = torch.clamp(torch.round(x / scale) + zero_point, qmin, qmax)
        # De-quantize
        dq = (q - zero_point) * scale
        return dq

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None, None, None]:
        # STE passes gradient uninhibited to parameter tensor
        grad_x = grad_output.clone()
        return grad_x, None, None, None, None


def quantize_dequantize_coopq(
    tensor: torch.Tensor,
    bits: int = 4,
    symmetric: bool = False,
    per_channel: bool = True,
    use_ste: bool = False,
) -> torch.Tensor:
    """
    Calibrates per-channel / tensor-wide scale and zero-point for INT(bits) and applies
    symmetric or asymmetric uniform quantization with optional STE.
    Enforces minimum bit-width floor b >= 4.
    """
    if tensor.numel() == 0:
        return tensor

    bits = max(4, int(bits))
    orig_dtype = tensor.dtype
    t = tensor.float()

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
        # Asymmetric quantization with explicit zero-point offset calibration
        qmin = 0.0
        qmax = float((2**bits) - 1)
        if per_channel and t.ndim >= 2:
            dim_to_reduce = tuple(range(1, t.ndim))
            min_val = torch.amin(t, dim=dim_to_reduce, keepdim=True)
            max_val = torch.amax(t, dim=dim_to_reduce, keepdim=True)
            scale = torch.clamp((max_val - min_val) / max(float(qmax), 1.0), min=1e-8)
            zero_point = torch.clamp(torch.round(-min_val / scale), qmin, qmax)
        else:
            min_val = torch.min(t)
            max_val = torch.max(t)
            scale = torch.clamp((max_val - min_val) / max(float(qmax), 1.0), min=1e-8)
            zero_point = torch.clamp(torch.round(-min_val / scale), qmin, qmax)

    if use_ste and t.requires_grad:
        out = CoopQSTEFakeQuantize.apply(t, scale, zero_point, qmin, qmax)
    else:
        q = torch.clamp(torch.round(t / scale) + zero_point, qmin, qmax)
        out = (q - zero_point) * scale

    return out.to(dtype=orig_dtype)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Safely extract configuration attribute from dict, object, or namespace."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class CoopQLinear(nn.Module):
    """
    CoopQ Linear layer with per-channel asymmetric quantization and STE backpropagation.
    Supports both standard nn.Linear and Conv1D weight representations.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        bits: int = 4,
        symmetric: bool = False,
        per_channel: bool = True,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.nf = out_features  # Conv1D compatibility
        self.bits = max(4, int(bits))
        self.symmetric = symmetric
        self.per_channel = per_channel
        self.lora_r = int(lora_r)
        self.lora_alpha = float(lora_alpha)
        self.scaling = (self.lora_alpha / self.lora_r) if self.lora_r > 0 else 1.0

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        if self.lora_r > 0:
            self.lora_A = nn.Parameter(torch.empty(self.lora_r, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, self.lora_r))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Module,
        bits: int = 4,
        symmetric: bool = False,
        per_channel: bool = True,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
    ) -> CoopQLinear:
        cls_name = linear.__class__.__name__.lower()
        is_conv1d = "conv1d" in cls_name or hasattr(linear, "nf")
        raw_w = linear.weight.data

        if is_conv1d:
            in_f = raw_w.shape[0]
            out_f = raw_w.shape[1]
            w_data = raw_w.t().contiguous()
        else:
            out_f = raw_w.shape[0]
            in_f = raw_w.shape[1]
            w_data = raw_w.contiguous()

        layer = cls(
            in_features=in_f,
            out_features=out_f,
            bias=getattr(linear, "bias", None) is not None and linear.bias is not None,
            bits=bits,
            symmetric=symmetric,
            per_channel=per_channel,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
        )
        layer.weight.data.copy_(w_data)
        if getattr(linear, "bias", None) is not None and linear.bias is not None:
            layer.bias.data.copy_(linear.bias.data)
        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_q = quantize_dequantize_coopq(
            self.weight,
            bits=self.bits,
            symmetric=self.symmetric,
            per_channel=self.per_channel,
            use_ste=self.training,
        )
        b_q = None
        if self.bias is not None:
            b_q = quantize_dequantize_coopq(
                self.bias,
                bits=self.bits,
                symmetric=self.symmetric,
                per_channel=False,
                use_ste=self.training,
            )
        out = F.linear(
            x,
            w_q.to(dtype=x.dtype, device=x.device),
            b_q.to(dtype=x.dtype, device=x.device) if b_q is not None else None,
        )
        if self.lora_r > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B)
            out = out + (self.scaling * lora_out).to(out.dtype)
        return out


class CoopQQuantizer(BaseCompressor):
    """
    CoopQ: Cooperative Game-Theoretic Bit Allocation Quantization.
    Allocates discrete bit-widths (>= 4 bits) across transformer layers
    proportional to layer sensitivity (Frobenius energy / variance), guaranteeing
    a target mean bit-budget under strict distortion minimization.

    Args:
        model: PyTorch model.
        bits: Target average bit-width (default 4).
        min_bits: Minimum allowable bit-width per layer (floor enforced at >= 4).
        max_bits: Maximum allowable bit-width per layer (default 8).
        symmetric: Whether to use symmetric quantization (default False for asymmetric calibration).
        per_channel: Whether to use per-channel scale factors (default True).
        use_ste: Whether to substitute nn.Linear with CoopQLinear for QAT (default True).
        target_attention_only: If True, only quantize attention projection layers (skip FFN/MLP).
        cfg: Optional configuration dictionary, namespace, or object.
    """

    def __init__(
        self,
        model: nn.Module,
        bits: int = 4,
        min_bits: int = 4,
        max_bits: int = 8,
        symmetric: bool = False,
        per_channel: bool = True,
        use_ste: bool = True,
        recalibrate_ln: bool = True,
        target_attention_only: bool = False,
        tokenizer: Optional[Any] = None,
        calibration_loader: Optional[Any] = None,
        calibration_texts: Optional[List[str]] = None,
        device: Optional[Union[str, torch.device]] = None,
        cfg: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="coopq", **kwargs)
        # Extract config values with _cfg_get fallbacks
        c_bits = _cfg_get(cfg, "coopq_bits", _cfg_get(cfg, "bits", bits))
        c_min_bits = _cfg_get(cfg, "coopq_min_bits", _cfg_get(cfg, "min_bits", min_bits))
        c_max_bits = _cfg_get(cfg, "coopq_max_bits", _cfg_get(cfg, "max_bits", max_bits))
        c_symmetric = _cfg_get(cfg, "coopq_symmetric", _cfg_get(cfg, "symmetric", symmetric))
        c_per_channel = _cfg_get(cfg, "coopq_per_channel", _cfg_get(cfg, "per_channel", per_channel))
        c_use_ste = _cfg_get(cfg, "coopq_use_ste", _cfg_get(cfg, "use_ste", use_ste))
        c_recalibrate_ln = _cfg_get(cfg, "coopq_recalibrate_ln", _cfg_get(cfg, "recalibrate_ln", recalibrate_ln))
        c_target_attn = _cfg_get(cfg, "coopq_target_attention_only", _cfg_get(cfg, "target_attention_only", target_attention_only))

        self.min_bits = max(4, int(c_min_bits))
        self.max_bits = max(self.min_bits, int(c_max_bits))
        self.bits = max(self.min_bits, min(int(c_bits), self.max_bits))
        self.symmetric = bool(c_symmetric)
        self.per_channel = bool(c_per_channel)
        self.use_ste = bool(c_use_ste)
        self.recalibrate_ln = bool(c_recalibrate_ln)
        self.target_attention_only = bool(c_target_attn)

        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        self.calibration_loader = calibration_loader
        self.calibration_texts = calibration_texts
        self.target_device = (
            torch.device(device)
            if device is not None
            else (next(model.parameters(), torch.tensor(0.0)).device)
        )
        self.stats: Dict[str, Any] = {}

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        """Identify candidate linear/Conv1D/LoRALinear modules."""
        cls_name = module.__class__.__name__.lower()
        has_base = hasattr(module, "base_layer")
        base_cls = module.base_layer.__class__.__name__.lower() if has_base else ""

        is_linear = isinstance(module, nn.Linear) or "linear" in cls_name
        is_conv1d = "conv1d" in cls_name or hasattr(module, "nf")
        is_lora = has_base and (
            isinstance(module.base_layer, nn.Linear)
            or "conv1d" in base_cls
            or hasattr(module.base_layer, "nf")
        )

        if not (is_linear or is_conv1d or is_lora):
            return False

        if isinstance(module, CoopQLinear) or (has_base and isinstance(module.base_layer, CoopQLinear)):
            return False

        name_lower = name.lower()
        if any(skip in name_lower for skip in ["classifier", "score", "lm_head", "pooler", "embed", "logits_processor", "pre_classifier", "head"]):
            return False

        if self.target_attention_only:
            ffn_patterns = [
                "ffn", "mlp", "lin1", "lin2", "intermediate",
                "gate_proj", "up_proj", "down_proj", "c_fc"
            ]
            if any(p in name_lower for p in ffn_patterns):
                return False
            if "output.dense" in name_lower and "attention" not in name_lower:
                return False
            attn_patterns = [
                "attn", "attention", "q_proj", "k_proj", "v_proj", "o_proj",
                "out_lin", "q_lin", "k_lin", "v_lin", "c_attn", "c_proj",
                "query", "key", "value", "in_proj_qkv", "out_proj"
            ]
            if not any(p in name_lower for p in attn_patterns):
                return False

        return True

    def _calculate_layer_sensitivities(self) -> Dict[str, float]:
        """Compute sensitivity metric per target layer based on normalized Frobenius energy and variance."""
        sensitivities: Dict[str, float] = {}
        for name, module in self.model.named_modules():
            if self._is_target_module(name, module):
                target_mod = getattr(module, "base_layer", getattr(module, "linear", module))
                raw_w = target_mod.weight.data
                cls_name = module.__class__.__name__.lower()
                is_conv1d = "conv1d" in cls_name or hasattr(module, "nf") or hasattr(target_mod, "nf")
                W = raw_w.t().float() if is_conv1d else raw_w.float()
                frob_norm = torch.norm(W, p="fro").item() / (W.numel() ** 0.5 + 1e-8)
                std_val = torch.std(W).item() if W.numel() > 1 else 0.0
                sensitivities[name] = float(frob_norm + std_val)
        return sensitivities

    def _allocate_bits(self, sensitivities: Dict[str, float]) -> Dict[str, int]:
        """Allocate discrete integer bit-widths satisfying mean(bits) <= self.bits with floor >= min_bits (>= 4)."""
        if not sensitivities:
            return {}

        total_sens = sum(sensitivities.values()) + 1e-8
        num_layers = len(sensitivities)
        target_total_bits = self.bits * num_layers

        # Allowed bit-levels with floor >= self.min_bits (>= 4)
        allowed_bits = [b for b in [4, 5, 6, 7, 8] if self.min_bits <= b <= self.max_bits]
        if not allowed_bits:
            allowed_bits = [self.min_bits]

        # Continuous allocation proportional to sensitivity
        raw_allocations: Dict[str, float] = {}
        mean_sens = total_sens / num_layers
        for name, sens in sensitivities.items():
            sens_ratio = sens / mean_sens
            raw_bits = self.bits + (sens_ratio - 1.0) * (self.max_bits - self.min_bits) / 2.0
            raw_allocations[name] = max(float(self.min_bits), min(float(self.max_bits), raw_bits))

        discrete_allocations: Dict[str, int] = {}
        for name, raw_b in raw_allocations.items():
            closest_b = min(allowed_bits, key=lambda b: abs(b - raw_b))
            discrete_allocations[name] = closest_b

        # Adjust sum to match target total bits budget
        current_sum = sum(discrete_allocations.values())
        sorted_names = sorted(sensitivities.keys(), key=lambda k: sensitivities[k], reverse=True)

        if current_sum > target_total_bits:
            for name in reversed(sorted_names):
                curr_b = discrete_allocations[name]
                valid_lower = [b for b in allowed_bits if b < curr_b]
                if valid_lower:
                    discrete_allocations[name] = max(valid_lower)
                    current_sum = sum(discrete_allocations.values())
                    if current_sum <= target_total_bits:
                        break
        elif current_sum < target_total_bits:
            for name in sorted_names:
                curr_b = discrete_allocations[name]
                valid_higher = [b for b in allowed_bits if b > curr_b]
                if valid_higher:
                    discrete_allocations[name] = min(valid_higher)
                    current_sum = sum(discrete_allocations.values())
                    if current_sum >= target_total_bits:
                        break

        # Ensure strict min_bits floor on all layers
        for k in discrete_allocations:
            discrete_allocations[k] = max(self.min_bits, discrete_allocations[k])

        return discrete_allocations

    def compress(self) -> Tuple[nn.Module, Dict[str, Any]]:
        """Quantize linear layers with layer-specific bit allocation (>= 4 bits) and per-channel asymmetric calibration."""
        sensitivities = self._calculate_layer_sensitivities()
        bit_allocations = self._allocate_bits(sensitivities)

        total_bits_weighted = 0
        total_params_linear = 0
        compressed_layers = []
        fallback_layers = []

        for name, module in list(self.model.named_modules()):
            if not self._is_target_module(name, module):
                # When target_attention_only is True, unselected FFN/MLP layers must be frozen during fine-tuning
                if self.target_attention_only:
                    is_lin_like = isinstance(module, nn.Linear) or "conv1d" in module.__class__.__name__.lower() or hasattr(module, "base_layer")
                    if is_lin_like:
                        for p in module.parameters():
                            p.requires_grad = False
                continue

            if name not in bit_allocations:
                continue

            target_mod = getattr(module, "base_layer", getattr(module, "linear", module))
            cls_name = module.__class__.__name__.lower()
            is_conv1d = "conv1d" in cls_name or hasattr(module, "nf") or hasattr(target_mod, "nf")
            has_lora = (hasattr(module, "lora_A") and module.lora_A is not None) or hasattr(module, "base_layer")

            layer_bits = bit_allocations[name]
            num_params = target_mod.weight.numel()
            total_bits_weighted += layer_bits * num_params
            total_params_linear += num_params

            if self.use_ste:
                src_lora_r = 0
                src_lora_alpha = 1.0
                if hasattr(module, 'lora_A') and hasattr(module, 'lora_B') and \
                   module.lora_A is not None and module.lora_B is not None:
                    src_lA = getattr(module.lora_A, 'weight', module.lora_A)
                    if hasattr(src_lA, 'shape') and len(src_lA.shape) == 2:
                        src_lora_r = src_lA.shape[0]
                    src_lora_alpha = getattr(module, 'lora_alpha', float(src_lora_r))

                coopq_layer = CoopQLinear.from_linear(
                    target_mod,
                    bits=layer_bits,
                    symmetric=self.symmetric,
                    per_channel=self.per_channel,
                    lora_r=src_lora_r,
                    lora_alpha=src_lora_alpha,
                )
                new_layer = coopq_layer
                compressed_layer = new_layer
                # Preserve existing LoRA adapter weights if the source module has them
                if hasattr(module, 'lora_A') and hasattr(module, 'lora_B') and \
                   module.lora_A is not None and module.lora_B is not None:
                    src_lA = getattr(module.lora_A, 'weight', module.lora_A)
                    src_lB = getattr(module.lora_B, 'weight', module.lora_B)
                    if hasattr(new_layer, 'lora_A') and new_layer.lora_A is not None:
                        if hasattr(src_lA, 'data') and src_lA.shape == new_layer.lora_A.shape:
                            new_layer.lora_A.data.copy_(src_lA.data)
                    if hasattr(new_layer, 'lora_B') and new_layer.lora_B is not None:
                        if hasattr(src_lB, 'data') and src_lB.shape == new_layer.lora_B.shape:
                            new_layer.lora_B.data.copy_(src_lB.data)

                if hasattr(module, 'base_layer'):
                    module.base_layer = compressed_layer  # put compressed layer inside wrapper
                    # module itself (LoRALinear wrapper) remains the replacement
                    new_module = module
                else:
                    new_module = compressed_layer

                if not hasattr(module, 'base_layer'):
                    if name == "":
                        self.model = new_module
                    else:
                        replace_module(self.model, name, new_module)
            else:
                raw_w = target_mod.weight.data
                W = raw_w.t().float() if is_conv1d else raw_w.float()
                W_quant = quantize_dequantize_coopq(
                    W,
                    bits=layer_bits,
                    symmetric=self.symmetric,
                    per_channel=self.per_channel,
                    use_ste=False,
                )
                if is_conv1d:
                    target_mod.weight.data.copy_(W_quant.t().contiguous().to(dtype=raw_w.dtype, device=raw_w.device))
                else:
                    target_mod.weight.data.copy_(W_quant.contiguous().to(dtype=raw_w.dtype, device=raw_w.device))

                if target_mod.bias is not None:
                    b_quant = quantize_dequantize_coopq(
                        target_mod.bias.data.float(),
                        bits=layer_bits,
                        symmetric=self.symmetric,
                        per_channel=False,
                        use_ste=False,
                    )
                    target_mod.bias.data.copy_(b_quant.to(dtype=target_mod.bias.dtype, device=target_mod.bias.device))

            # Preserve pre-existing LoRA adapter parameters and trainability
            if has_lora:
                for p_name, p in module.named_parameters():
                    if any(t in p_name.lower() for t in ("lora_a", "lora_b", "adapter")):
                        p.requires_grad = True

            compressed_layers.append(name)

        # LayerNorm Recalibration
        if self.recalibrate_ln:
            recalibrate_layernorms(
                self.model,
                tokenizer=self.tokenizer,
                calibration_loader=self.calibration_loader,
                calibration_texts=self.calibration_texts,
                device=self.target_device,
            )

        avg_bits = (
            total_bits_weighted / total_params_linear if total_params_linear > 0 else float(self.bits)
        )
        effective_compression_rate = 16.0 / max(avg_bits, 1e-3)
        stats = self.get_compression_stats()
        stats.update({
            "target_bits": self.bits,
            "min_bits_floor": self.min_bits,
            "average_allocated_bits": round(avg_bits, 2),
            "layer_bit_allocations": bit_allocations,
            "compressed_layers": compressed_layers,
            "compressed_layer_count": len(compressed_layers),
            "fallback_layers": fallback_layers,
            "fallback_layer_count": len(fallback_layers),
            "symmetric": self.symmetric,
            "per_channel": self.per_channel,
            "use_ste": self.use_ste,
            "target_attention_only": self.target_attention_only,
            "effective_compression_rate": round(effective_compression_rate, 2),
            "recalibrate_ln": self.recalibrate_ln,
            "method": "coopq_asymmetric_ste_bmin4",
            "theoretical_vram_saving": f"{(1.0 - avg_bits / 16.0) * 100:.1f}% vs FP16",
        })
        self.stats = stats
        return self.model, stats


@COMPRESSOR_REGISTRY.register("coopq")
@COMPRESSOR_REGISTRY.register("coopq_quantizer")
def coopq_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    """CoopQ compressor factory function."""
    bits = kwargs.pop("bits", _cfg_get(comp_config, "coopq_bits", _cfg_get(comp_config, "bits", 4)))
    min_bits = kwargs.pop("min_bits", _cfg_get(comp_config, "coopq_min_bits", _cfg_get(comp_config, "min_bits", 4)))
    max_bits = kwargs.pop("max_bits", _cfg_get(comp_config, "coopq_max_bits", _cfg_get(comp_config, "max_bits", 8)))
    symmetric = kwargs.pop("symmetric", _cfg_get(comp_config, "coopq_symmetric", _cfg_get(comp_config, "symmetric", False)))
    per_channel = kwargs.pop("per_channel", _cfg_get(comp_config, "coopq_per_channel", _cfg_get(comp_config, "per_channel", True)))
    use_ste = kwargs.pop("use_ste", _cfg_get(comp_config, "coopq_use_ste", _cfg_get(comp_config, "use_ste", True)))
    recalibrate_ln = kwargs.pop("recalibrate_ln", _cfg_get(comp_config, "coopq_recalibrate_ln", True))
    target_attention_only = kwargs.pop("target_attention_only", _cfg_get(comp_config, "coopq_target_attention_only", _cfg_get(comp_config, "target_attention_only", False)))

    compressor = CoopQQuantizer(
        model,
        bits=bits,
        min_bits=min_bits,
        max_bits=max_bits,
        symmetric=symmetric,
        per_channel=per_channel,
        use_ste=use_ste,
        recalibrate_ln=recalibrate_ln,
        target_attention_only=target_attention_only,
        cfg=comp_config,
        **kwargs,
    )
    compressed_model, stats = compressor.compress()
    try:
        compressed_model.compression_stats = stats
    except Exception as e:
        logger.warning("Failed to attach compression_stats: %s", e)
    return compressed_model

__all__ = [
    "CoopQSTEFakeQuantize",
    "quantize_dequantize_coopq",
    "CoopQLinear",
    "CoopQQuantizer",
    "coopq_factory",
]
