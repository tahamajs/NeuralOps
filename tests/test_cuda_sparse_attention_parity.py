# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Parity tests for Block Sparse Attention (Module 6).
"""

import pytest
import torch

from neuralops.core.cuda_kernels.sparse_attention import (
    block_sparse_attn,
    block_sparse_attn_pytorch_ref,
)


def test_block_sparse_attn_parity():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for Block Sparse Attention test")

    torch.manual_seed(42)
    B, H, N, D = 2, 4, 32, 64
    block_size = 16
    num_blocks = N // block_size

    q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)
    k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)
    v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)

    block_mask = torch.eye(num_blocks, device="cuda", dtype=torch.int32)
    block_mask[0, 1] = 1

    out_cuda = block_sparse_attn(q, k, v, block_mask, block_size=block_size)
    out_ref = block_sparse_attn_pytorch_ref(q, k, v, block_mask, block_size=block_size)

    torch.testing.assert_close(out_cuda, out_ref, atol=1e-3, rtol=1e-3)
