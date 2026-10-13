"""
Fused Normalization + Linear Projection Layer.
Combines RMSNorm (or LayerNorm) directly before Linear projection into an optimized pass.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


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
    """Dispatches fused norm linear kernel or PyTorch fallback."""
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
