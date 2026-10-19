#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Block-Sparse Attention Autograd Interface

Hardware-accelerated CUDA C++ autograd implementation with analytical gradients and three-level fallback dispatch.
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
