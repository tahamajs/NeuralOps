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
        x, scale, zero_point = ctx.saved_tensors
        qmin = ctx.qmin
        qmax = ctx.qmax

        # Scaled values in quantization grid
        scaled_x = (x / scale) + zero_point
        # Zero out gradients outside quantization range
        mask = (scaled_x >= qmin) & (scaled_x <= qmax)
        grad_x = grad_output * mask.to(dtype=grad_output.dtype)

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
            zero_point = torch.round(-min_val / scale)
        else:
            min_val = torch.min(t)
            max_val = torch.max(t)
            scale = torch.clamp((max_val - min_val) / max(float(qmax), 1.0), min=1e-8)
            zero_point = torch.round(-min_val / scale)

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
        linear: nn.Linear,
        bits: int = 8,
        symmetric: bool = True,
        per_channel: bool = False,
    ) -> AstroLinear:
        layer = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            bits=bits,
            symmetric=symmetric,
            per_channel=per_channel,
        )
        layer.weight.data.copy_(linear.weight.data)
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
        return nn.functional.linear(x, w_q, b_q)


class AstroQuantizer(BaseCompressor):
    """
    AstroQuantizer: Uniform min-max / symmetric quantization engine for linear weights and biases.

    Args:
        model: PyTorch model.
        bits: Bit-width for quantization (e.g. 4 or 8). Default 8.
        symmetric: Whether to use symmetric quantization around 0. Default True.
        per_channel: Whether to use per-channel scale factors. Default False.
        use_ste: Whether to substitute nn.Linear with AstroLinear for QAT with STE. Default False.
    """

    def __init__(
        self,
        model: nn.Module,
        bits: int = 8,
        symmetric: bool = True,
        per_channel: bool = False,
        use_ste: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="astro", **kwargs)
        self.bits = int(bits)
        self.symmetric = bool(symmetric)
        self.per_channel = bool(per_channel)
        self.use_ste = bool(use_ste)

    def _replace_module(self, full_name: str, new_module: nn.Module) -> None:
        parts = full_name.split(".")
        parent = self.model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_module)

    def compress(self) -> Tuple[nn.Module, Dict[str, Any]]:
        """Quantize all linear layers in the model and return (compressed_model, stats)."""
        for name, module in list(self.model.named_modules()):
            if isinstance(module, nn.Linear):
                if self.use_ste:
                    astro_layer = AstroLinear.from_linear(
                        module,
                        bits=self.bits,
                        symmetric=self.symmetric,
                        per_channel=self.per_channel,
                    )
                    self._replace_module(name, astro_layer)
                else:
                    module.weight.data = quantize_dequantize_tensor(
                        module.weight.data,
                        bits=self.bits,
                        symmetric=self.symmetric,
                        per_channel=self.per_channel,
                        use_ste=False,
                    )
                    if module.bias is not None:
                        module.bias.data = quantize_dequantize_tensor(
                            module.bias.data,
                            bits=self.bits,
                            symmetric=self.symmetric,
                            per_channel=False,
                            use_ste=False,
                        )

        stats = self.get_compression_stats()
        stats["quantization_bits"] = self.bits
        stats["symmetric"] = self.symmetric
        stats["per_channel"] = self.per_channel
        stats["theoretical_vram_saving"] = f"{(1.0 - self.bits / 16.0) * 100:.1f}% vs FP16"
        return self.model, stats


@COMPRESSOR_REGISTRY.register("astro")
def astro_factory(model: nn.Module, comp_config: Any = None) -> nn.Module:
    """Astro compressor factory function."""
    bits = getattr(comp_config, "astro_bits", 8) if comp_config else 8
    symmetric = getattr(comp_config, "astro_symmetric", True) if comp_config else True
    per_channel = getattr(comp_config, "astro_per_channel", False) if comp_config else False
    use_ste = getattr(comp_config, "astro_use_ste", False) if comp_config else False

    compressor = AstroQuantizer(
        model,
        bits=bits,
        symmetric=symmetric,
        per_channel=per_channel,
        use_ste=use_ste,
    )
    compressed_model, _ = compressor.compress()
    return compressed_model
