"""
Numerical parity tests between Triton kernels and PyTorch fallbacks across precisions.
"""

import pytest
import torch

from neuralops.core.triton_kernels._common import (
    linear_attn_pytorch_fallback,
    sparse_attn_pytorch_fallback,
    low_rank_pytorch_fallback,
    state_update_pytorch_fallback,
    gqa_pytorch_fallback,
    triton_dispatch,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_linear_attn_parity(dtype):
    B, H, N, D = 2, 4, 32, 64
    q = torch.randn(B, H, N, D, dtype=dtype)
    k = torch.randn(B, H, N, D, dtype=dtype)
    v = torch.randn(B, H, N, D, dtype=dtype)

    out_fallback = linear_attn_pytorch_fallback(q, k, v)
    out_dispatch = triton_dispatch("linear_attn", q, k, v)

    # In CPU/fallback environments, dispatch routes to fallback; in CUDA it routes to Triton.
    # Outputs should match tightly.
    torch.testing.assert_close(out_dispatch, out_fallback, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_low_rank_parity(dtype):
    B, N, in_dim, rank, out_dim = 2, 16, 64, 16, 64
    x = torch.randn(B, N, in_dim, dtype=dtype)
    U = torch.randn(in_dim, rank, dtype=dtype)
    V = torch.randn(rank, out_dim, dtype=dtype)
    bias = torch.randn(out_dim, dtype=dtype)

    out_fallback = low_rank_pytorch_fallback(x, U, V, bias=bias)
    out_dispatch = triton_dispatch("low_rank", x, U, V, bias=bias)

    torch.testing.assert_close(out_dispatch, out_fallback, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", [torch.float32])
def test_gqa_parity(dtype):
    B, num_kv_heads, N, D = 2, 2, 16, 32
    num_heads = 4
    k = torch.randn(B, num_kv_heads, N, D, dtype=dtype)
    v = torch.randn(B, num_kv_heads, N, D, dtype=dtype)

    k_f, v_f = gqa_pytorch_fallback(k, v, num_kv_heads=num_kv_heads, num_heads=num_heads)
    k_d, v_d = triton_dispatch("gqa", k, v, num_kv_heads=num_kv_heads, num_heads=num_heads)

    torch.testing.assert_close(k_d, k_f)
    torch.testing.assert_close(v_d, v_f)


def test_sparse_attn_fallback_shapes():
    B, H, N, D = 1, 2, 16, 32
    q = torch.randn(B, H, N, D)
    k = torch.randn(B, H, N, D)
    v = torch.randn(B, H, N, D)

    out = sparse_attn_pytorch_fallback(q, k, v)
    assert out.shape == q.shape
