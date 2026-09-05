"""FAQ (Feature-Activation-Aware Quantization): adjusts quantization scales using activation statistics. Registered as 'faq'."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from framework.compression_engine.astro_quantizer import (
    STEFakeQuantize,
    quantize_dequantize_tensor,
)
from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY


class FAQLinear(nn.Module):
    """Linear layer using FAQ activation-aware quantization with STE backpropagation and optional LoRA adapter."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        bits: int = 8,
        symmetric: bool = True,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.nf = out_features  # Conv1D compatibility
        self.bits = bits
        self.symmetric = symmetric
        self.lora_r = int(lora_r)
        self.lora_alpha = float(lora_alpha)
        self.scaling = (self.lora_alpha / self.lora_r) if self.lora_r > 0 else 1.0

        requires_grad = not (self.lora_r > 0)
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype),
            requires_grad=requires_grad,
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, device=device, dtype=dtype),
                requires_grad=requires_grad,
            )
        else:
            self.register_parameter("bias", None)
        self.register_buffer("act_scale", torch.ones(in_features, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

        if self.lora_r > 0:
            self.lora_A = nn.Parameter(
                torch.empty((self.lora_r, in_features), dtype=torch.float32, device=device)
            )
            self.lora_B = nn.Parameter(
                torch.zeros((out_features, self.lora_r), dtype=torch.float32, device=device)
            )
            nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    @classmethod
    def from_linear(
        cls,
        linear: Union[nn.Linear, Any],
        act_scale: torch.Tensor,
        bits: int = 8,
        symmetric: bool = True,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
    ) -> FAQLinear:
        cls_name = linear.__class__.__name__.lower()
        is_conv1d = "conv1d" in cls_name or hasattr(linear, "nf")
        raw_w = linear.weight.data
        if is_conv1d:
            in_f, out_f = raw_w.shape[0], raw_w.shape[1]
            W = raw_w.t()
        else:
            out_f, in_f = raw_w.shape[0], raw_w.shape[1]
            W = raw_w

        layer = cls(
            in_features=in_f,
            out_features=out_f,
            bias=getattr(linear, "bias", None) is not None and linear.bias is not None,
            bits=bits,
            symmetric=symmetric,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            device=W.device,
            dtype=W.dtype,
        )
        layer.weight.data.copy_(W)
        if getattr(linear, "bias", None) is not None and linear.bias is not None:
            layer.bias.data.copy_(linear.bias.data)
        layer.act_scale.copy_(act_scale.view(-1)[:in_f].to(device=W.device, dtype=W.dtype))
        return layer

    def get_weight(self) -> torch.Tensor:
        s = torch.clamp(self.act_scale, min=1e-6).unsqueeze(0).to(device=self.weight.device, dtype=self.weight.dtype)
        w_scaled = self.weight * s
        w_q = quantize_dequantize_tensor(
            w_scaled,
            bits=self.bits,
            symmetric=self.symmetric,
            per_channel=True,
            use_ste=False,
        )
        return w_q / s

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Scale weights by activation sensitivity for quantization
        s = torch.clamp(self.act_scale, min=1e-6).unsqueeze(0).to(device=self.weight.device, dtype=self.weight.dtype)
        w_scaled = self.weight * s
        w_q = quantize_dequantize_tensor(
            w_scaled,
            bits=self.bits,
            symmetric=self.symmetric,
            per_channel=True,
            use_ste=self.training,
        )
        w_restored = w_q / s

        b_q = None
        if self.bias is not None:
            b_q = quantize_dequantize_tensor(
                self.bias,
                bits=self.bits,
                symmetric=self.symmetric,
                per_channel=False,
                use_ste=self.training,
            )
        out = nn.functional.linear(
            x,
            w_restored.to(dtype=x.dtype, device=x.device),
            b_q.to(dtype=x.dtype, device=x.device) if b_q is not None else None,
        )

        if self.lora_r > 0 and self.lora_A is not None and self.lora_B is not None:
            out = out + self.scaling * nn.functional.linear(
                nn.functional.linear(x.float(), self.lora_A), self.lora_B
            ).to(out.dtype)

        return out


class FAQQuantizer(BaseCompressor):
    """
    FAQQuantizer: Feature-Activation-Aware Quantization.
    Collects activation statistics to scale channel-wise quantization step sizes,
    minimizing activation-weighted reconstruction distortion.

    Args:
        model: PyTorch model.
        tokenizer: Optional tokenizer for text-based calibration.
        bits: Quantization bit-width (4 or 8). Default 8.
        calibration_texts: List of text strings for calibration.
        calibration_data: Optional Tensor or DataLoader for calibration.
        alpha: Activation scaling exponent. Default 0.5.
        symmetric: Whether to use symmetric quantization. Default True.
        use_ste: Whether to substitute nn.Linear with FAQLinear. Default False.
        lora_r: Optional LoRA rank.
        lora_alpha: LoRA scaling alpha.
        max_error: Maximum reconstruction error before fallback.
        fallback_to_original: If True, keep original layer on error excess.
        skip_attention_qkv: Skip asymmetric GQA projections.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Optional[Any] = None,
        bits: int = 8,
        calibration_texts: Optional[List[str]] = None,
        calibration_data: Optional[Any] = None,
        alpha: float = 0.5,
        symmetric: bool = True,
        use_ste: bool = False,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        max_error: float = 0.25,
        fallback_to_original: bool = True,
        skip_attention_qkv: bool = True,
        cfg: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="faq", **kwargs)
        self.tokenizer = tokenizer
        self.bits = int(bits or getattr(cfg, "faq_bits", 8))
        self.calibration_texts = calibration_texts
        self.calibration_data = calibration_data
        self.alpha = float(alpha if alpha is not None else getattr(cfg, "faq_alpha", 0.5))
        self.symmetric = bool(symmetric if symmetric is not None else getattr(cfg, "faq_symmetric", True))
        self.use_ste = bool(use_ste if use_ste is not None else getattr(cfg, "faq_use_ste", False))
        self.lora_r = int(getattr(cfg, "lora_r", getattr(cfg, "lora_rank", lora_r)))
        self.lora_alpha = float(getattr(cfg, "lora_alpha", lora_alpha))
        self.max_error = float(getattr(cfg, "max_error", max_error))
        self.fallback_to_original = bool(getattr(cfg, "fallback_to_original", fallback_to_original))
        self.skip_attention_qkv = bool(getattr(cfg, "skip_attention_qkv", skip_attention_qkv))
        self.cfg = cfg

        self.compressed_layers: List[str] = []
        self.fallback_layers: List[str] = []
        self.skipped_layers: List[str] = []
        self.layer_errors: Dict[str, float] = {}

    def _is_gqa_asymmetric(self, name: str, module: nn.Module) -> bool:
        if not self.skip_attention_qkv:
            return False
        cfg = getattr(self.model, "config", None)
        num_heads = getattr(cfg, "num_attention_heads", None)
        num_kv_heads = getattr(cfg, "num_key_value_heads", None)
        if num_heads is not None and num_kv_heads is not None and num_kv_heads < num_heads:
            name_lower = name.lower()
            if any(p in name_lower for p in ["q_proj", "k_proj", "v_proj", "query", "key", "value", "c_attn"]):
                return True
        return False

    def _collect_activation_scales(self) -> Dict[str, torch.Tensor]:
        """Collect channel-wise input activation scales via empirical calibration hooks with running covariance."""
        running_cov: Dict[str, torch.Tensor] = {}
        sample_counts: Dict[str, int] = {}
        hooks = []

        def make_hook(layer_name: str) -> Any:
            def hook(module: nn.Module, inp: Tuple[torch.Tensor, ...], outp: Any) -> None:
                if len(inp) > 0 and isinstance(inp[0], torch.Tensor):
                    x = inp[0].detach().float()
                    cls_name = module.__class__.__name__.lower()
                    is_conv1d = "conv1d" in cls_name or hasattr(module, "nf")
                    in_f = module.weight.shape[0] if is_conv1d else module.weight.shape[1]
                    if x.ndim >= 2:
                        flat_x = x.reshape(-1, in_f)
                    else:
                        flat_x = x.unsqueeze(0)
                    n_samples = flat_x.shape[0]
                    if n_samples > 0:
                        batch_cov = torch.matmul(flat_x.T, flat_x).cpu()
                        if layer_name not in running_cov:
                            running_cov[layer_name] = batch_cov
                            sample_counts[layer_name] = n_samples
                        else:
                            running_cov[layer_name] += batch_cov
                            sample_counts[layer_name] += n_samples

            return hook

        from framework.utils.compat import should_compress_linear

        for name, module in self.model.named_modules():
            cls_name = module.__class__.__name__.lower()
            is_linear_like = isinstance(module, nn.Linear) or "conv1d" in cls_name or hasattr(module, "nf")
            if is_linear_like and should_compress_linear(self.model, name) and not self._is_gqa_asymmetric(name, module):
                hooks.append(module.register_forward_hook(make_hook(name)))

        self.model.eval()
        with torch.no_grad():
            try:
                if self.tokenizer is not None and (self.calibration_texts or hasattr(self.model, "tokenizer")):
                    texts = self.calibration_texts or ["The quick brown fox jumps over the lazy dog"]
                    inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
                    first_p = next(self.model.parameters(), None)
                    if first_p is not None and first_p.is_cuda:
                        inputs = {k: v.cuda() for k, v in inputs.items()}
                    self.model(**inputs)
                elif self.calibration_data is not None:
                    if isinstance(self.calibration_data, torch.Tensor):
                        first_p = next(self.model.parameters(), None)
                        dev = first_p.device if first_p is not None else torch.device("cpu")
                        self.model(self.calibration_data.to(dev))
                    elif hasattr(self.calibration_data, "__iter__"):
                        for idx, batch in enumerate(self.calibration_data):
                            if isinstance(batch, torch.Tensor):
                                first_p = next(self.model.parameters(), None)
                                dev = first_p.device if first_p is not None else torch.device("cpu")
                                self.model(batch.to(dev))
                            elif isinstance(batch, dict):
                                self.model(**batch)
                            if idx >= 32:
                                break
                else:
                    # Synthetic calibration forward pass
                    first_linear = next((m for m in self.model.modules() if isinstance(m, nn.Linear)), None)
                    if first_linear is not None:
                        first_p = next(self.model.parameters(), None)
                        dev = first_p.device if first_p is not None else torch.device("cpu")
                        dummy_x = torch.randn(2, 16, first_linear.in_features, device=dev)
                        try:
                            self.model(dummy_x)
                        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, OSError, ImportError):
                            pass
            except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, OSError, ImportError):
                pass

        for h in hooks:
            h.remove()

        act_scales: Dict[str, torch.Tensor] = {}
        for name, module in self.model.named_modules():
            cls_name = module.__class__.__name__.lower()
            is_linear_like = isinstance(module, nn.Linear) or "conv1d" in cls_name or hasattr(module, "nf")
            if is_linear_like and hasattr(module, "weight"):
                is_conv1d = "conv1d" in cls_name or hasattr(module, "nf")
                in_f = module.weight.shape[0] if is_conv1d else module.weight.shape[1]
                if name in running_cov and sample_counts.get(name, 0) > 0:
                    cov = running_cov[name] / float(max(sample_counts[name], 1))
                    var_diag = torch.diagonal(cov)
                    channel_scale = torch.sqrt(torch.clamp(var_diag, min=1e-6))
                    act_scales[name] = channel_scale
                else:
                    act_scales[name] = torch.ones(in_f)

        return act_scales

    def compress(self) -> Tuple[nn.Module, Dict[str, Any]]:
        """Quantize model weights using FAQ activation statistics."""
        from framework.core.model_utils import replace_module
        from framework.utils.compat import should_compress_linear

        act_scales = self._collect_activation_scales()
        self.compressed_layers = []
        self.fallback_layers = []
        self.skipped_layers = []
        self.layer_errors = {}
        seen_weights: set[int] = set()

        for name, module in list(self.model.named_modules()):
            cls_name = module.__class__.__name__.lower()
            is_linear_like = isinstance(module, nn.Linear) or "conv1d" in cls_name or hasattr(module, "nf")
            if not is_linear_like or isinstance(module, FAQLinear):
                continue

            if not should_compress_linear(self.model, name):
                continue

            if self._is_gqa_asymmetric(name, module):
                continue

            if not hasattr(module, "weight") or module.weight is None:
                continue

            w_id = id(module.weight)
            if w_id in seen_weights:
                self.skipped_layers.append(name)
                continue
            seen_weights.add(w_id)

            raw_w = module.weight.data
            is_conv1d = "conv1d" in cls_name or hasattr(module, "nf")
            W_orig = raw_w.t().float() if is_conv1d else raw_w.float()
            in_f = W_orig.shape[1]

            s_act = act_scales.get(name, torch.ones(in_f)).to(raw_w.device)
            s_pow = torch.pow(torch.clamp(s_act, min=1e-6), self.alpha).unsqueeze(0)

            try:
                if self.use_ste or self.lora_r > 0:
                    faq_layer = FAQLinear.from_linear(
                        module,
                        act_scale=s_act,
                        bits=self.bits,
                        symmetric=self.symmetric,
                        lora_r=self.lora_r,
                        lora_alpha=self.lora_alpha,
                    )
                    W_recon = faq_layer.get_weight().float()
                    W_norm = torch.norm(W_orig, p="fro").clamp_min(1e-8)
                    err = float(torch.norm(W_orig - W_recon, p="fro").item() / W_norm)
                    self.layer_errors[name] = err

                    if err > self.max_error:
                        if self.fallback_to_original:
                            self.fallback_layers.append(name)
                            continue

                    replace_module(self.model, name, faq_layer)
                    self.compressed_layers.append(name)
                else:
                    W_scaled = W_orig * s_pow
                    W_q = quantize_dequantize_tensor(
                        W_scaled,
                        bits=self.bits,
                        symmetric=self.symmetric,
                        per_channel=True,
                        use_ste=False,
                    )
                    W_recon = W_q / s_pow
                    W_norm = torch.norm(W_orig, p="fro").clamp_min(1e-8)
                    err = float(torch.norm(W_orig - W_recon, p="fro").item() / W_norm)
                    self.layer_errors[name] = err

                    if err > self.max_error:
                        if self.fallback_to_original:
                            self.fallback_layers.append(name)
                            continue

                    if is_conv1d:
                        module.weight.data.copy_(W_recon.t().to(dtype=raw_w.dtype))
                    else:
                        module.weight.data.copy_(W_recon.to(dtype=raw_w.dtype))

                    if getattr(module, "bias", None) is not None and module.bias is not None:
                        module.bias.data = quantize_dequantize_tensor(
                            module.bias.data,
                            bits=self.bits,
                            symmetric=self.symmetric,
                            per_channel=False,
                            use_ste=False,
                        )
                    self.compressed_layers.append(name)
            except Exception as e:
                self.fallback_layers.append(name)

        # Post-compression forward sanity check
        forward_check_success = True
        try:
            with torch.no_grad():
                dev = next(self.model.parameters(), torch.tensor(0.0)).device
                dummy = torch.randint(0, 100, (1, 8), device=dev)
                try:
                    self.model(dummy)
                except Exception:
                    first_linear = next(
                        (m for m in self.model.modules() if isinstance(m, (nn.Linear, FAQLinear))),
                        None,
                    )
                    in_dim = getattr(first_linear, "in_features", 8)
                    self.model(torch.randn(1, 8, in_dim, device=dev))
        except Exception as e:
            forward_check_success = False

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        avg_err = (sum(self.layer_errors.values()) / len(self.layer_errors)) if self.layer_errors else 0.0
        stats.update(
            {
                "compressed_layers": self.compressed_layers,
                "compressed_layer_count": len(self.compressed_layers),
                "fallback_layers": self.fallback_layers,
                "fallback_layer_count": len(self.fallback_layers),
                "skipped_layers": self.skipped_layers,
                "skipped_layer_count": len(self.skipped_layers),
                "layer_errors": self.layer_errors,
                "avg_reconstruction_error": avg_err,
                "quantization_bits": self.bits,
                "alpha": self.alpha,
                "symmetric": self.symmetric,
                "lora_r": self.lora_r,
                "lora_alpha": self.lora_alpha,
                "forward_check_success": forward_check_success,
                "theoretical_vram_saving": f"{(1.0 - self.bits / 16.0) * 100:.1f}% vs FP16",
            }
        )
        return self.model, stats


@COMPRESSOR_REGISTRY.register("faq_factory")
@COMPRESSOR_REGISTRY.register("faq_quantizer_factory")
def faq_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    """FAQ compressor factory function."""
    bits = kwargs.pop("bits", getattr(comp_config, "faq_bits", 8) if comp_config is not None else 8)
    alpha = kwargs.pop("alpha", getattr(comp_config, "faq_alpha", 0.5) if comp_config is not None else 0.5)
    symmetric = kwargs.pop("symmetric", getattr(comp_config, "faq_symmetric", True) if comp_config is not None else True)
    use_ste = kwargs.pop("use_ste", getattr(comp_config, "faq_use_ste", False) if comp_config is not None else False)
    lora_r = kwargs.pop("lora_r", getattr(comp_config, "lora_r", 0) if comp_config is not None else 0)
    lora_alpha = kwargs.pop("lora_alpha", getattr(comp_config, "lora_alpha", 1.0) if comp_config is not None else 1.0)
    tokenizer = kwargs.pop("tokenizer", getattr(model, "tokenizer", None))
    return_stats = kwargs.pop("return_stats", kwargs.pop("return_tuple", False))

    compressor = FAQQuantizer(
        model,
        tokenizer=tokenizer,
        bits=bits,
        alpha=alpha,
        symmetric=symmetric,
        use_ste=use_ste,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        cfg=comp_config,
        **kwargs,
    )
    compressed_model, stats = compressor.compress()
    if return_stats:
        return compressed_model, stats
    return compressed_model
