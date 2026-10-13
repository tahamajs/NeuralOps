"""
Numerical parity tests between Triton kernels and PyTorch fallbacks across precisions.
Tests both CPU fallback paths and forced CUDA Triton kernel execution when CUDA is present.
"""

import pytest
import torch
import torch.nn.functional as F

from neuralops.core.triton_kernels import (
    HAS_TRITON,
    fms_scan,
    fms_scan_triton,
    gdn_scan,
    gdn_scan_triton,
    gla_scan,
    gla_scan_triton,
    gqa,
    gqa_pytorch_fallback,
    hgrn2_scan,
    hgrn2_scan_triton,
    linear_attn,
    linear_attn_pytorch_fallback,
    low_rank,
    low_rank_pytorch_fallback,
    sparse_attn,
    sparse_attn_pytorch_fallback,
    state_update,
    state_update_pytorch_fallback,
    triton_dispatch,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_linear_attn_parity(dtype):
    B, H, N, D = 2, 4, 32, 64
    device = "cuda" if torch.cuda.is_available() else "cpu"
    q = torch.randn(B, H, N, D, dtype=dtype, device=device)
    k = torch.randn(B, H, N, D, dtype=dtype, device=device)
    v = torch.randn(B, H, N, D, dtype=dtype, device=device)

    out_fallback = linear_attn_pytorch_fallback(q, k, v)
    out_dispatch = triton_dispatch("linear_attn", q, k, v)

    torch.testing.assert_close(out_dispatch, out_fallback, rtol=1e-2, atol=1e-2)

    # If CUDA & Triton available, explicitly verify triton kernel fwd against fallback
    if device == "cuda" and HAS_TRITON and linear_attn.is_supported(q, k, v):
        out_triton = linear_attn.fwd(q, k, v)
        torch.testing.assert_close(out_triton, out_fallback, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_low_rank_parity(dtype):
    B, N, in_dim, rank, out_dim = 2, 16, 64, 16, 64
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(B, N, in_dim, dtype=dtype, device=device)
    U = torch.randn(in_dim, rank, dtype=dtype, device=device)
    V = torch.randn(rank, out_dim, dtype=dtype, device=device)
    bias = torch.randn(out_dim, dtype=dtype, device=device)

    out_fallback = low_rank_pytorch_fallback(x, U, V, bias=bias)
    out_dispatch = triton_dispatch("low_rank", x, U, V, bias=bias)

    torch.testing.assert_close(out_dispatch, out_fallback, rtol=1e-2, atol=1e-2)

    if device == "cuda" and HAS_TRITON and low_rank.is_supported(x, U, V, bias=bias):
        out_triton = low_rank.fwd(x, U, V, bias=bias)
        torch.testing.assert_close(out_triton, out_fallback, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_gqa_parity(dtype):
    B, num_kv_heads, N, D = 2, 2, 16, 32
    num_heads = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"
    k = torch.randn(B, num_kv_heads, N, D, dtype=dtype, device=device)
    v = torch.randn(B, num_kv_heads, N, D, dtype=dtype, device=device)

    k_f, v_f = gqa_pytorch_fallback(k, v, num_kv_heads=num_kv_heads, num_heads=num_heads)
    k_d, v_d = triton_dispatch("gqa", k, v, num_kv_heads=num_kv_heads, num_heads=num_heads)

    torch.testing.assert_close(k_d, k_f)
    torch.testing.assert_close(v_d, v_f)

    if device == "cuda" and HAS_TRITON and gqa.is_supported(k, v, num_kv_heads, num_heads):
        k_t, v_t = gqa.fwd(k, v, num_kv_heads, num_heads)
        torch.testing.assert_close(k_t, k_f)
        torch.testing.assert_close(v_t, v_f)


def test_sparse_attn_fallback_shapes():
    B, H, N, D = 1, 2, 16, 32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    q = torch.randn(B, H, N, D, device=device)
    k = torch.randn(B, H, N, D, device=device)
    v = torch.randn(B, H, N, D, device=device)

    out = sparse_attn_pytorch_fallback(q, k, v)
    assert out.shape == q.shape
    out_d = triton_dispatch("sparse_attn", q, k, v)
    assert out_d.shape == q.shape


@pytest.mark.skipif(not torch.cuda.is_available() or not HAS_TRITON, reason="CUDA and Triton required")
def test_scan_kernels_cuda_execution():
    """Verify recurrent scan kernels directly on CUDA with Triton."""
    B, H, L, D = 2, 4, 32, 32
    device = "cuda"

    # 1. GDN Scan
    if gdn_scan_triton is not None:
        q = torch.randn(B, H, L, D, device=device, dtype=torch.float32)
        k = F.normalize(torch.randn(B, H, L, D, device=device, dtype=torch.float32), dim=-1)
        v = torch.randn(B, H, L, D, device=device, dtype=torch.float32)
        beta = torch.sigmoid(torch.randn(B, H, L, D, device=device, dtype=torch.float32))
        alpha = torch.sigmoid(torch.randn(B, H, L, D, device=device, dtype=torch.float32))
        out_gdn = gdn_scan_triton(q, k, v, beta, alpha)
        assert out_gdn.shape == q.shape

    # 2. HGRN2 Scan
    if hgrn2_scan_triton is not None:
        f = torch.randn(B, H, L, D, device=device, dtype=torch.float32)
        iv = torch.randn(B, H, L, D, device=device, dtype=torch.float32)
        out_hgrn2 = hgrn2_scan_triton(f, iv, res_clip=1.0)
        assert out_hgrn2.shape == f.shape
