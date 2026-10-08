"""
Bridge module for neuralops.core.triton_sparse_attn
Re-exports Triton-accelerated block-sparse attention kernel from canonical sparse_attn.py.
"""
from typing import Any
import torch

from neuralops.core.triton_kernels.sparse_attn import fwd as _sparse_attn_fwd
from neuralops.core.triton_kernels._common import HAS_TRITON


def triton_block_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_mask: torch.Tensor,
    block_size: int = 64,
    token_mask: torch.Tensor | None = None,
    is_causal: bool = False,
    **kwargs: Any,
) -> torch.Tensor:
    """Wrapper function delegating to canonical sparse_attn kernel."""
    return _sparse_attn_fwd(
        q,
        k,
        v,
        block_mask=block_mask,
        block_size=block_size,
        is_causal=is_causal,
        token_mask=token_mask,
        **kwargs,
    )


__all__ = ["triton_block_sparse_attention", "HAS_TRITON"]

