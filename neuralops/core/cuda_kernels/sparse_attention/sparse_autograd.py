"""
Sparse Attention Python Interface & Autograd Backward (Compatibility Wrapper).
"""

from neuralops.core.cuda_kernels.sparse_attention.sparse_attention_autograd import (
    _get_module,
    block_sparse_attn_pytorch_ref,
    block_sparse_attn,
    block_sparse_attn_bwd,
    _BlockSparseAttnAutograd,
)

__all__ = [
    "_get_module",
    "block_sparse_attn_pytorch_ref",
    "block_sparse_attn",
    "block_sparse_attn_bwd",
    "_BlockSparseAttnAutograd",
]
