"""
Bridge module for neuralops.core.triton_linear_attn
Re-exports Triton-accelerated linear attention kernel from canonical linear_attn.py.
"""
from typing import Any
import torch

from neuralops.core.triton_kernels.linear_attn import fwd as _linear_attn_fwd
from neuralops.core.triton_kernels._common import HAS_TRITON


def fast_linear_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = True,
    eps: float = 1e-6,
    **kwargs: Any,
) -> torch.Tensor:
    """Canonical wrapper for fast fused linear attention."""
    return _linear_attn_fwd(q, k, v, is_causal=is_causal, eps=eps, **kwargs)


class LinearAttentionTriton:
    @staticmethod
    def apply(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = True, eps: float = 1e-6) -> torch.Tensor:
        return fast_linear_attention(q, k, v, is_causal=is_causal, eps=eps)


__all__ = ["fast_linear_attention", "LinearAttentionTriton", "HAS_TRITON"]

