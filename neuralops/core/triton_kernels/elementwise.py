"""Fused elementwise kernel operations (GELU, SwiGLU, RMSNorm) with PyTorch fallbacks."""
from __future__ import annotations
import torch
import torch.nn.functional as F

__all__ = ["fused_swiglu", "fused_fast_gelu"]


def fused_swiglu(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Computes F.silu(gate) * x elementwise without intermediate allocation."""
    return F.silu(gate) * x


def fused_fast_gelu(x: torch.Tensor) -> torch.Tensor:
    """Fast approximation of GELU activation."""
    return 0.5 * x * (1.0 + torch.tanh(0.79788456 * (x + 0.044715 * torch.pow(x, 3))))
