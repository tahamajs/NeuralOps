"""
Comprehensive GPU tests for PagedAttention (Module 1).
Validates warp-register execution, dynamic fallback kernel, FP16/BF16 & INT64 safety,
and end-to-end autograd backward parity.
"""

import pytest
import torch

from neuralops.core.cuda_kernels.paged_attention import (
    paged_attention,
    paged_attention_pytorch_reference,
)


def _build_paged_tensors(num_seqs, num_heads, num_kv_heads, head_dim, block_size, max_blocks, dtype, int_dtype):
    num_total_blocks = 64
    q = torch.randn(num_seqs, num_heads, head_dim, device="cuda", dtype=dtype)
    k_cache = torch.randn(num_total_blocks, num_kv_heads, head_dim, block_size, device="cuda", dtype=dtype)
    v_cache = torch.randn(num_total_blocks, num_kv_heads, head_dim, block_size, device="cuda", dtype=dtype)

    context_lens = torch.randint(1, max_blocks * block_size + 1, (num_seqs,), device="cuda", dtype=int_dtype)
    context_lens[0] = 0  # Include edge case of empty sequence

    # Non-contiguous scattered block tables
    perm = torch.randperm(num_total_blocks, device="cuda", dtype=int_dtype)
    block_tables = perm[:num_seqs * max_blocks].view(num_seqs, max_blocks)

    return q, k_cache, v_cache, block_tables, context_lens


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("int_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("head_dim,block_size", [(32, 8), (64, 16), (128, 32)])
@pytest.mark.parametrize("num_heads,num_kv_heads", [(8, 8), (8, 2)])
def test_paged_attention_standard_shapes(dtype, int_dtype, head_dim, block_size, num_heads, num_kv_heads):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    torch.manual_seed(42)
    num_seqs = 4
    max_blocks = 4

    q, k_cache, v_cache, block_tables, context_lens = _build_paged_tensors(
        num_seqs, num_heads, num_kv_heads, head_dim, block_size, max_blocks, dtype, int_dtype
    )

    out_cuda = paged_attention(q, k_cache, v_cache, block_tables, context_lens, block_size)
    out_ref = paged_attention_pytorch_reference(q, k_cache, v_cache, block_tables, context_lens, block_size)

    tol = 1e-2 if dtype == torch.bfloat16 else 1e-3
    torch.testing.assert_close(out_cuda, out_ref, atol=tol, rtol=tol)
    # Check that empty sequence at index 0 produces exact zeros
    assert torch.all(out_cuda[0] == 0.0)


def test_paged_attention_dynamic_fallback_uncommon_shape():
    """Tests non-standard head_dim=48 and block_size=12 to verify paged_attention_dynamic_kernel."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    torch.manual_seed(42)
    head_dim = 48
    block_size = 12
    num_seqs = 4
    num_heads = 6
    num_kv_heads = 2
    max_blocks = 5

    q, k_cache, v_cache, block_tables, context_lens = _build_paged_tensors(
        num_seqs, num_heads, num_kv_heads, head_dim, block_size, max_blocks, torch.float32, torch.int32
    )

    out_cuda = paged_attention(q, k_cache, v_cache, block_tables, context_lens, block_size)
    out_ref = paged_attention_pytorch_reference(q, k_cache, v_cache, block_tables, context_lens, block_size)

    torch.testing.assert_close(out_cuda, out_ref, atol=1e-3, rtol=1e-3)


def test_paged_attention_non_contiguous_views():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    torch.manual_seed(42)
    raw_q = torch.randn(8, 2, 64, device="cuda", dtype=torch.float32)
    q = raw_q.transpose(0, 1)  # [2, 8, 64] non-contiguous view

    k_cache = torch.randn(16, 2, 64, 16, device="cuda", dtype=torch.float32)
    v_cache = torch.randn(16, 2, 64, 16, device="cuda", dtype=torch.float32)
    block_tables = torch.arange(8, device="cuda", dtype=torch.int64).view(2, 4)
    context_lens = torch.tensor([32, 48], device="cuda", dtype=torch.int64)

    out_cuda = paged_attention(q, k_cache, v_cache, block_tables, context_lens, 16)
    out_ref = paged_attention_pytorch_reference(q, k_cache, v_cache, block_tables, context_lens, 16)

    torch.testing.assert_close(out_cuda, out_ref, atol=1e-3, rtol=1e-3)


def test_paged_attention_autograd_backward():
    """Verifies that calling .backward() on paged_attention computes exact gradients."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    torch.manual_seed(42)
    num_seqs = 2
    num_heads = 4
    num_kv_heads = 2
    head_dim = 64
    block_size = 16
    num_blocks = 16

    # Clone identical inputs for CUDA and PyTorch Reference
    q1 = torch.randn(num_seqs, num_heads, head_dim, device="cuda", dtype=torch.float32, requires_grad=True)
    k1 = torch.randn(num_blocks, num_kv_heads, head_dim, block_size, device="cuda", dtype=torch.float32, requires_grad=True)
    v1 = torch.randn(num_blocks, num_kv_heads, head_dim, block_size, device="cuda", dtype=torch.float32, requires_grad=True)

    q2 = q1.clone().detach().requires_grad_(True)
    k2 = k1.clone().detach().requires_grad_(True)
    v2 = v1.clone().detach().requires_grad_(True)

    block_tables = torch.tensor([[0, 2], [1, 3]], device="cuda", dtype=torch.int32)
    context_lens = torch.tensor([24, 30], device="cuda", dtype=torch.int32)

    # CUDA backward
    out1 = paged_attention(q1, k1, v1, block_tables, context_lens, block_size)
    loss1 = (out1 ** 2).sum()
    loss1.backward()

    # PyTorch reference backward
    out2 = paged_attention_pytorch_reference(q2, k2, v2, block_tables, context_lens, block_size)
    loss2 = (out2 ** 2).sum()
    loss2.backward()

    torch.testing.assert_close(out1, out2, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(q1.grad, q2.grad, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(k1.grad, k2.grad, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(v1.grad, v2.grad, atol=1e-3, rtol=1e-3)
