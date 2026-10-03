# framework/utils/layer_norm_fix.py
"""
Numerically Stable Layer Normalization and RMS Normalization Modules.

Eliminates FP16 variance underflow and sum-of-squares overflow by forcing all
mean, variance, and reduction accumulators into torch.float32 prior to reciprocal square root.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm) with FP32 accumulator precision.

    In standard FP16 mixed-precision training, squaring high-magnitude activations
    can exceed the FP16 maximum value (65504), causing overflow to `inf` / `nan`, or
    underflow to zero for small activations. This implementation explicitly casts
    the reduction accumulators to `torch.float32` before computing the sum-of-squares
    and reciprocal square root:
        $$y = \\frac{x}{\\sqrt{\\frac{1}{d} \\sum_{i=1}^d x_i^2 + \\epsilon}} \\odot \\gamma$$
    """

    def __init__(
        self,
        normalized_shape: int | tuple[int, ...] | list[int] | torch.Size,
        eps: float = 1e-6,
        elementwise_affine: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if isinstance(normalized_shape, int):
            self.normalized_shape = (normalized_shape,)
        else:
            self.normalized_shape = tuple(normalized_shape)
        self.eps: float = float(eps)
        self.elementwise_affine: bool = elementwise_affine

        if elementwise_affine:
            self.weight = nn.Parameter(
                torch.ones(self.normalized_shape, device=device, dtype=dtype)
            )
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with FP32 accumulator computation.

        Args:
            x: Input tensor of shape `(..., *normalized_shape)`.

        Returns:
            Normalized tensor with matching input dtype and shape.
        """
        input_dtype = x.dtype
        reduction_dims = tuple(range(-len(self.normalized_shape), 0))

        # FIX: Explicitly cast accumulators to torch.float32 to avoid FP16 sum-of-squares overflow/underflow
        variance = torch.mean(x.float() ** 2, dim=reduction_dims, keepdim=True)
        x_normed = (x.float() * torch.rsqrt(variance + self.eps)).to(input_dtype)

        if self.weight is not None:
            x_normed = x_normed * self.weight.to(input_dtype)
        return x_normed.to(input_dtype)

    def extra_repr(self) -> str:
        return f"{self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine}"


class LayerNorm(nn.Module):
    """
    Standard Layer Normalization with FP32 accumulator precision.

    Prevents FP16 numerical instability and variance underflow by computing
    mean and centered variance in `torch.float32`:
        $$y = \\frac{x - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}} \\odot \\gamma + \\beta$$
    """

    def __init__(
        self,
        normalized_shape: int | tuple[int, ...] | list[int] | torch.Size,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if isinstance(normalized_shape, int):
            self.normalized_shape = (normalized_shape,)
        else:
            self.normalized_shape = tuple(normalized_shape)
        self.eps: float = float(eps)
        self.elementwise_affine: bool = elementwise_affine

        if elementwise_affine:
            self.weight = nn.Parameter(
                torch.ones(self.normalized_shape, device=device, dtype=dtype)
            )
            if bias:
                self.bias = nn.Parameter(
                    torch.zeros(self.normalized_shape, device=device, dtype=dtype)
                )
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with FP32 accumulator computation.

        Args:
            x: Input tensor of shape `(..., *normalized_shape)`.

        Returns:
            Normalized tensor with matching input dtype and shape.
        """
        input_dtype = x.dtype
        reduction_dims = tuple(range(-len(self.normalized_shape), 0))

        # FIX: Explicitly cast accumulators to torch.float32 for mean and variance to prevent FP16 variance underflow
        x_fp32 = x.float()
        mean = torch.mean(x_fp32, dim=reduction_dims, keepdim=True)
        variance = torch.mean((x_fp32 - mean) ** 2, dim=reduction_dims, keepdim=True)
        x_normed = ((x_fp32 - mean) * torch.rsqrt(variance + self.eps)).to(input_dtype)

        if self.weight is not None:
            x_normed = x_normed * self.weight.to(input_dtype)
        if self.bias is not None:
            x_normed = x_normed + self.bias.to(input_dtype)
        return x_normed.to(input_dtype)

    def extra_repr(self) -> str:
        return f"{self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine}"


# Aliases for explicit usage
SafeRMSNorm = RMSNorm
SafeLayerNorm = LayerNorm
FP32AccumulatorRMSNorm = RMSNorm
FP32AccumulatorLayerNorm = LayerNorm


def fix_layernorm_naming(model: nn.Module) -> None:
    """
    Fix LayerNorm parameter naming (e.g. HuggingFace gamma/beta to weight/bias).
    """
    for module in model.modules():
        if isinstance(module, (nn.LayerNorm, LayerNorm)):
            state = dict(module.named_parameters())
            if "beta" in state and "bias" not in state:
                bias = module._parameters.pop("beta")
                module.register_parameter("bias", bias)
            if "gamma" in state and "weight" not in state:
                weight = module._parameters.pop("gamma")
                module.register_parameter("weight", weight)


def replace_layernorm_with_safe_norm(model: nn.Module) -> nn.Module:
    """
    Recursively replaces all standard PyTorch nn.LayerNorm instances in a model
    with FP32 accumulator SafeLayerNorm instances.
    """
    for name, child in model.named_children():
        if isinstance(child, nn.LayerNorm) and not isinstance(child, LayerNorm):
            has_bias = child.bias is not None
            safe_ln = LayerNorm(
                normalized_shape=child.normalized_shape,
                eps=child.eps,
                elementwise_affine=child.elementwise_affine,
                bias=has_bias,
            )
            if child.elementwise_affine:
                with torch.no_grad():
                    if child.weight is not None:
                        safe_ln.weight.copy_(child.weight)
                    if has_bias and child.bias is not None:
                        safe_ln.bias.copy_(child.bias)
            setattr(model, name, safe_ln)
        else:
            replace_layernorm_with_safe_norm(child)
    return model
