"""
Bridge module for neuralops.core.triton_sparse_attn
Re-exports Triton-accelerated block-sparse attention kernel.
"""
from neuralops.core.triton_kernels.triton_sparse_attn import (
    triton_block_sparse_attention,
    HAS_TRITON,
)

__all__ = ["triton_block_sparse_attention", "HAS_TRITON"]
