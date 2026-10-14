"""
Fused Normalization + Linear Projection Layer with Autograd Backward.
Combines RMSNorm (or LayerNorm) directly before Linear projection into an optimized pass.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class _FusedNormLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor], eps: float):
        ctx.eps = eps
        ctx.has_bias = bias is not None
        orig_shape = x.shape
        D = orig_shape[-1]
        x_2d = x.reshape(-1, D)

        # LayerNorm forward: mean & var
        mean = x_2d.mean(dim=-1, keepdim=True)
        var = ((x_2d - mean) ** 2).mean(dim=-1, keepdim=True)
        rstd = torch.rsqrt(var + eps)
        x_hat = (x_2d - mean) * rstd

        # Linear projection
        out_2d = F.linear(x_hat, weight, bias)
        out = out_2d.reshape(*orig_shape[:-1], weight.shape[0])

        ctx.save_for_backward(x_2d, mean, rstd, x_hat, weight)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x_2d, mean, rstd, x_hat, weight = ctx.saved_tensors
        D = x_2d.shape[-1]
        grad_out_2d = grad_output.reshape(-1, weight.shape[0])

        # dweight = grad_out^T @ x_hat
        dweight = torch.matmul(grad_out_2d.t(), x_hat)
        dbias = grad_out_2d.sum(dim=0) if ctx.has_bias else None

        # dx_hat = grad_out @ weight
        dx_hat = torch.matmul(grad_out_2d, weight)

        # LayerNorm backward
        # dx = rstd * (dx_hat - mean(dx_hat) - x_hat * mean(dx_hat * x_hat))
        mean_dx_hat = dx_hat.mean(dim=-1, keepdim=True)
        mean_dx_hat_x_hat = (dx_hat * x_hat).mean(dim=-1, keepdim=True)
        dx_2d = rstd * (dx_hat - mean_dx_hat - x_hat * mean_dx_hat_x_hat)

        dx = dx_2d.reshape(ctx.saved_tensors[0].shape)
        # Match original input shape
        if grad_output.dim() > 2:
            dx = dx.reshape(*grad_output.shape[:-1], D)

        return dx, dweight, dbias, None


def fused_norm_linear_pytorch(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Reference PyTorch implementation of fused LayerNorm + Linear."""
    normed = F.layer_norm(x, (x.shape[-1],), eps=eps)
    return F.linear(normed, weight, bias)


def fused_norm_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Dispatches fused norm linear kernel with full autograd backward."""
    if torch.is_grad_enabled() or x.requires_grad or weight.requires_grad:
        return _FusedNormLinearFunction.apply(x, weight, bias, eps)
    return fused_norm_linear_pytorch(x, weight, bias=bias, eps=eps)


class FusedRMSNormLinear(nn.Module):
    """
    Combines RMSNorm followed by nn.Linear.
    """

    def __init__(self, in_features: int, out_features: int, eps: float = 1e-6, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        self.norm = nn.RMSNorm(in_features, eps=eps)
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm(x))


__all__ = [
    "FusedRMSNormLinear",
    "fused_norm_linear",
    "fused_norm_linear_pytorch",
]
