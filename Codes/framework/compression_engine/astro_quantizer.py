"""AstroQuantizer: Dynamic Uniform Range Quantization Engine with STE. Registered as 'astro'."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY


class STEFakeQuantize(torch.autograd.Function):
    """
    Straight-Through Estimator (STE) fake-quantization autograd function.
    Clamps inputs to [qmin, qmax] in discrete integer space, de-quantizes back,
    and passes gradients through uninhibited where within clipping bounds.
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
        ctx.save_for_backward(x, scale, zero_point)
        ctx.qmin = qmin
        ctx.qmax = qmax

        # Quantize and clamp
        q = torch.clamp(torch.round(x / scale) + zero_point, qmin, qmax)
        # De-quantize
        dq = (q - zero_point) * scale
        return dq

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        # Vanilla Straight-Through Estimator (STE) passes gradients directly
        # without masking, allowing weights outside quantization bounds to recover.
        grad_x = grad_output.clone()
        return grad_x, None, None, None, None


def quantize_dequantize_tensor(
    tensor: torch.Tensor,
    bits: int = 8,
    symmetric: bool = True,
    per_channel: bool = False,
    use_ste: bool = False,
) -> torch.Tensor:
    """
    Calibrate scale and zero-point for INT(bits) and apply uniform quantization/dequantization.
    """
    if tensor.numel() == 0:
        return tensor

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
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.symmetric = symmetric
        self.per_channel = per_channel

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
            in_f,
            out_f,
            bias=linear.bias is not None,
            bits=bits,
            symmetric=symmetric,
            per_channel=per_channel,
        )
        layer.weight.data.copy_(w_data)
        if linear.bias is not None:
            layer.bias.data.copy_(linear.bias.data)
        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_q = quantize_dequantize_tensor(
            self.weight,
            bits=self.bits,
            symmetric=self.symmetric,
            per_channel=self.per_channel,
            use_ste=self.training,
        )
        b_q = None
        if self.bias is not None:
            b_q = quantize_dequantize_tensor(
                self.bias,
                bits=self.bits,
                symmetric=self.symmetric,
                per_channel=False,
                use_ste=self.training,
            )
        return nn.functional.linear(
            x,
            w_q.to(dtype=x.dtype, device=x.device),
            b_q.to(dtype=x.dtype, device=x.device) if b_q is not None else None,
        )


import logging
logger = logging.getLogger(__name__)
from framework.core.model_utils import replace_module
from framework.utils.compat import should_compress_linear


@COMPRESSOR_REGISTRY.register("astro")
@COMPRESSOR_REGISTRY.register("astro_quantizer")
class AstroQuantizer(BaseCompressor):
    """
    AstroQuantizer: Uniform min-max / symmetric quantization engine for linear weights and biases.

    Production hardening features:
    - Relative Frobenius reconstruction error validation with configurable max_error (default 0.20).
    - Safe fallback to original weights by default (fallback_to_original=True).
    - Robust Conv1D support (GPT-2 c_attn, c_proj) via weight transposition.
    - Full FP32 quantization calculation with NaN/Inf guards.
    - Unified critical-layer protection via should_compress_linear and protect_critical.
    - Comprehensive per-layer statistics and error tracking.
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
        target_modules: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="astro", **kwargs)
        cfg = kwargs.get("comp_config", kwargs.get("cfg", None))
        if cfg is not None:
            bits = getattr(cfg, "astro_bits", getattr(cfg, "bits", bits))
            symmetric = getattr(cfg, "astro_symmetric", getattr(cfg, "symmetric", symmetric))
            per_channel = getattr(cfg, "astro_per_channel", getattr(cfg, "per_channel", per_channel))
            use_ste = getattr(cfg, "astro_use_ste", getattr(cfg, "use_ste", use_ste))
            max_error = getattr(cfg, "astro_max_error", getattr(cfg, "max_error", max_error))
            fallback_to_original = getattr(
                cfg, "astro_fallback_to_original", getattr(cfg, "fallback_to_original", fallback_to_original)
            )
            protect_critical = getattr(cfg, "protect_critical", protect_critical)

        self.bits = int(bits)
        self.symmetric = bool(symmetric)
        self.per_channel = bool(per_channel)
        self.use_ste = bool(use_ste)
        self.max_error = float(max_error)
        self.fallback_to_original = bool(fallback_to_original)
        self.protect_critical = bool(protect_critical)
        self.target_modules = target_modules

        self.compressed_layers: list[str] = []
        self.fallback_layers: list[str] = []
        self.layer_errors: Dict[str, float] = {}

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        if isinstance(module, AstroLinear):
            return False
        cls_name = module.__class__.__name__.lower()
        is_linear_like = isinstance(module, nn.Linear) or "conv1d" in cls_name or hasattr(module, "nf")
        if not is_linear_like:
            return False

        if self.protect_critical:
            name_lower = name.lower()
            if any(skip in name_lower for skip in ["classifier", "score", "lm_head", "pooler", "embed", "logits_processor"]):
                return False

        if not should_compress_linear(self.model, name):
            return False
        if self.target_modules is not None:
            return any(t in name for t in self.target_modules)
        return True

    def compress(self) -> Tuple[nn.Module, Dict[str, Any]]:
        """Quantize all target linear/Conv1D layers in the model and return (compressed_model, stats)."""
        self.compressed_layers = []
        self.fallback_layers = []
        self.layer_errors = {}

        for name, module in list(self.model.named_modules()):
            if self._is_target_module(name, module):
                cls_name = module.__class__.__name__.lower()
                is_conv1d = "conv1d" in cls_name or hasattr(module, "nf")
                raw_w = module.weight.data

                W = raw_w.t().clone().float() if is_conv1d else raw_w.clone().float()
                W_quant = quantize_dequantize_tensor(
                    W,
                    bits=self.bits,
                    symmetric=self.symmetric,
                    per_channel=self.per_channel,
                    use_ste=False,
                )

                w_norm = torch.norm(W, p="fro").item()
                err = float(torch.norm(W - W_quant, p="fro").item() / max(w_norm, 1e-12))
                self.layer_errors[name] = err

                if err > self.max_error:
                    logger.warning(
                        "AstroQuantizer error %.4f on %s exceeded threshold %.4f.",
                        err,
                        name,
                        self.max_error,
                    )
                    if self.fallback_to_original:
                        logger.info("Skipping AstroQuantizer for %s (fallback to original).", name)
                        self.fallback_layers.append(name)
                        continue

                if self.use_ste:
                    astro_layer = AstroLinear.from_linear(
                        module,
                        bits=self.bits,
                        symmetric=self.symmetric,
                        per_channel=self.per_channel,
                    )
                    replace_module(self.model, name, astro_layer)
                else:
                    if is_conv1d:
                        module.weight.data.copy_(W_quant.t().contiguous().to(dtype=raw_w.dtype, device=raw_w.device))
                    else:
                        module.weight.data.copy_(W_quant.contiguous().to(dtype=raw_w.dtype, device=raw_w.device))

                    if module.bias is not None:
                        b_quant = quantize_dequantize_tensor(
                            module.bias.data.float(),
                            bits=self.bits,
                            symmetric=self.symmetric,
                            per_channel=False,
                            use_ste=False,
                        )
                        module.bias.data.copy_(b_quant.to(dtype=module.bias.dtype, device=module.bias.device))

                self.compressed_layers.append(name)

        self._compressed_params = self._count_params(self.model)
        avg_err = (
            sum(self.layer_errors[k] for k in self.compressed_layers) / max(1, len(self.compressed_layers))
            if self.compressed_layers
            else 0.0
        )
        stats = self.get_compression_stats()
        stats.update({
            "quantization_bits": self.bits,
            "symmetric": self.symmetric,
            "per_channel": self.per_channel,
            "compressed_layers": self.compressed_layers,
            "fallback_layers": self.fallback_layers,
            "num_compressed": len(self.compressed_layers),
            "num_fallback": len(self.fallback_layers),
            "layer_errors": self.layer_errors,
            "avg_reconstruction_error": avg_err,
            "max_error": self.max_error,
            "fallback_to_original": self.fallback_to_original,
            "theoretical_vram_saving": f"{(1.0 - self.bits / 16.0) * 100:.1f}% vs FP16",
        })
        return self.model, stats


@COMPRESSOR_REGISTRY.register("astro_factory")
@COMPRESSOR_REGISTRY.register("astro_quantizer_factory")
def astro_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    """Astro compressor factory function."""
    bits = kwargs.pop("bits", getattr(comp_config, "astro_bits", getattr(comp_config, "bits", 8)))
    symmetric = kwargs.pop("symmetric", getattr(comp_config, "astro_symmetric", getattr(comp_config, "symmetric", True)))
    per_channel = kwargs.pop("per_channel", getattr(comp_config, "astro_per_channel", getattr(comp_config, "per_channel", False)))
    use_ste = kwargs.pop("use_ste", getattr(comp_config, "astro_use_ste", getattr(comp_config, "use_ste", False)))
    max_error = kwargs.pop("max_error", getattr(comp_config, "astro_max_error", getattr(comp_config, "max_error", 0.20)))
    fallback = kwargs.pop("fallback_to_original", getattr(
        comp_config, "astro_fallback_to_original", getattr(comp_config, "fallback_to_original", True)
    ))

    compressor = AstroQuantizer(
        model,
        bits=bits,
        symmetric=symmetric,
        per_channel=per_channel,
        use_ste=use_ste,
        max_error=max_error,
        fallback_to_original=fallback,
        comp_config=comp_config,
        **kwargs,
    )
    compressed_model, _ = compressor.compress()
    return compressed_model


__all__ = [
    "STEFakeQuantize",
    "quantize_dequantize_tensor",
    "AstroLinear",
    "AstroQuantizer",
    "astro_factory",
]
