# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Triton Sparse Attention Bridge

Backward-compatibility bridge exposing Triton-accelerated block-sparse attention kernels.
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

