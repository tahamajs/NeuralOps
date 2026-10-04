"""
Bridge module for neuralops.core.triton_linear_attn
Re-exports Triton-accelerated linear attention kernel.
"""
from neuralops.core.triton_kernels.triton_linear_attn import (
    fast_linear_attention,
    LinearAttentionTriton,
    HAS_TRITON,
)

__all__ = ["fast_linear_attention", "LinearAttentionTriton", "HAS_TRITON"]
