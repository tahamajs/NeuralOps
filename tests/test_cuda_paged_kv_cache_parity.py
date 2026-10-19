#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Parity tests for Paged KV Cache block allocation and CoW copy (Module 8).
"""

import pytest
import torch

from neuralops.core.cuda_kernels.paged_kv_cache import (
    copy_blocks_cow,
    copy_blocks_cow_pytorch_ref,
)


def test_kv_cache_cow_copy_parity():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for Paged KV Cache test")

    torch.manual_seed(42)
    num_blocks = 16
    block_size = 16
    num_heads = 8
    head_dim = 64

    src_k = torch.randn(num_blocks, num_heads, head_dim, block_size, device="cuda", dtype=torch.float32)
    src_v = torch.randn(num_blocks, num_heads, head_dim, block_size, device="cuda", dtype=torch.float32)

    dst_k_cuda = torch.zeros_like(src_k)
    dst_v_cuda = torch.zeros_like(src_v)

    dst_k_ref = torch.zeros_like(src_k)
    dst_v_ref = torch.zeros_like(src_v)

    block_mapping = torch.tensor([[0, 2], [1, 5], [3, 7]], device="cuda", dtype=torch.int32)

    copy_blocks_cow(dst_k_cuda, dst_v_cuda, src_k, src_v, block_mapping, block_size, num_heads, head_dim)
    copy_blocks_cow_pytorch_ref(dst_k_ref, dst_v_ref, src_k, src_v, block_mapping, block_size, num_heads, head_dim)

    torch.testing.assert_close(dst_k_cuda, dst_k_ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(dst_v_cuda, dst_v_ref, atol=1e-5, rtol=1e-5)
