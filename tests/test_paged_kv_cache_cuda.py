# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Comprehensive GPU tests for Paged KV Cache (Module 8).
Validates 3-level dispatch, float32/float16/bfloat16 safety, int64/int32 mapping,
and negative/empty block guards.
"""

import pytest
import torch

from neuralops.core.cuda_kernels.paged_kv_cache import copy_blocks_cow, copy_blocks_cow_pytorch_ref


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("block_mapping_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("num_heads,head_dim,block_size", [(4, 32, 8), (8, 64, 16), (16, 128, 32)])
def test_paged_kv_cache_dtypes_and_shapes(dtype, block_mapping_dtype, num_heads, head_dim, block_size):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    torch.manual_seed(42)
    num_blocks = 32
    src_k = torch.randn(num_blocks, num_heads, block_size, head_dim, device="cuda", dtype=dtype)
    src_v = torch.randn(num_blocks, num_heads, block_size, head_dim, device="cuda", dtype=dtype)

    dst_k_cuda = torch.zeros_like(src_k)
    dst_v_cuda = torch.zeros_like(src_v)
    dst_k_ref = torch.zeros_like(src_k)
    dst_v_ref = torch.zeros_like(src_v)

    mapping = torch.tensor([[0, 5], [1, 7], [3, 12], [4, 15], [8, 2], [9, 6], [10, 11], [14, 20]],
                           device="cuda", dtype=block_mapping_dtype)

    copy_blocks_cow(dst_k_cuda, dst_v_cuda, src_k, src_v, mapping, block_size, num_heads, head_dim)
    copy_blocks_cow_pytorch_ref(dst_k_ref, dst_v_ref, src_k, src_v, mapping, block_size, num_heads, head_dim)

    torch.testing.assert_close(dst_k_cuda, dst_k_ref)
    torch.testing.assert_close(dst_v_cuda, dst_v_ref)


def test_paged_kv_cache_empty_and_negative_indices():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    torch.manual_seed(42)
    num_blocks = 16
    src_k = torch.randn(num_blocks, 8, 16, 64, device="cuda", dtype=torch.float32)
    src_v = torch.randn(num_blocks, 8, 16, 64, device="cuda", dtype=torch.float32)

    dst_k = torch.zeros_like(src_k)
    dst_v = torch.zeros_like(src_v)
    empty_mapping = torch.empty((0, 2), device="cuda", dtype=torch.int64)

    copy_blocks_cow(dst_k, dst_v, src_k, src_v, empty_mapping, 16, 8, 64)
    assert torch.all(dst_k == 0.0)
    assert torch.all(dst_v == 0.0)

    dst_k_ref = torch.zeros_like(src_k)
    dst_v_ref = torch.zeros_like(src_v)
    bad_mapping = torch.tensor([[-1, 2], [3, -1], [999, 1], [2, 999], [0, 4]], device="cuda", dtype=torch.int64)

    copy_blocks_cow(dst_k, dst_v, src_k, src_v, bad_mapping, 16, 8, 64)
    copy_blocks_cow_pytorch_ref(dst_k_ref, dst_v_ref, src_k, src_v, bad_mapping, 16, 8, 64)

    torch.testing.assert_close(dst_k, dst_k_ref)
    torch.testing.assert_close(dst_v, dst_v_ref)
    torch.testing.assert_close(dst_k[4], src_k[0])
    torch.testing.assert_close(dst_v[4], src_v[0])


def test_paged_kv_cache_non_contiguous():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    num_blocks = 8
    raw_src_k = torch.randn(num_blocks, 16, 8, 64, device="cuda", dtype=torch.float32)
    raw_src_v = torch.randn(num_blocks, 16, 8, 64, device="cuda", dtype=torch.float32)

    src_k = raw_src_k.transpose(1, 2)
    src_v = raw_src_v.transpose(1, 2)
    dst_k = torch.zeros_like(src_k)
    dst_v = torch.zeros_like(src_v)

    mapping = torch.tensor([[1, 3], [2, 5]], device="cuda", dtype=torch.int32)
    copy_blocks_cow(dst_k, dst_v, src_k, src_v, mapping, 16, 8, 64)

    torch.testing.assert_close(dst_k[3], src_k[1])
    torch.testing.assert_close(dst_v[5], src_v[2])
