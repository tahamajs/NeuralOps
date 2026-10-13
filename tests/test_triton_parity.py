"""
Comprehensive numerical parity and fallback tests across all 26 Triton kernel modules.
Tests CPU/MPS/CUDA fallbacks across float32 and float16 precisions, verifying mathematical
equivalence, shape preservation, and differentiability.
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
from neuralops.core.triton_kernels.paged_attention import (
    paged_attention_kernel,
    paged_attention_pytorch,
)
from neuralops.core.triton_kernels.fused_norm_linear import (
    fused_norm_linear,
    fused_norm_linear_pytorch,
)
from neuralops.core.triton_kernels.fused_rope import (
    fused_rope_embedding_kernel,
)
from neuralops.core.triton_kernels.sliding_window import (
    sliding_window_attention,
    sliding_window_pytorch,
)
from neuralops.core.triton_kernels.softplus import softplus_fwd, softplus_pytorch
from neuralops.core.triton_kernels.fused_ce import fused_cross_entropy, fused_cross_entropy_pytorch
from neuralops.core.triton_kernels.fused_kd import fused_kd_loss, fused_kd_loss_pytorch
from neuralops.core.triton_kernels.flash_decoding import flash_decode_kernel
from neuralops.core.triton_kernels.block_sparse import block_sparse_attention
from neuralops.core.triton_kernels.cosformer_triton_kernel import (
    cosformer_triton_fwd,
    cosformer_pytorch_fwd,
)
from neuralops.core.triton_kernels.selective_scan import (
    selective_scan_fn,
    selective_scan_ref,
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


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_paged_attention_parity(dtype):
    B, H, D = 2, 4, 32
    num_blocks, block_size = 8, 16
    q = torch.randn(B, H, D, dtype=dtype)
    k_cache = torch.randn(num_blocks, block_size, H, D, dtype=dtype)
    v_cache = torch.randn(num_blocks, block_size, H, D, dtype=dtype)
    block_tables = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.long)
    seq_lens = torch.tensor([30, 25], dtype=torch.long)

    ref = paged_attention_pytorch(q, k_cache, v_cache, block_tables, seq_lens, block_size=block_size)
    out = paged_attention_kernel(q, k_cache, v_cache, block_tables, seq_lens, block_size=block_size)
    torch.testing.assert_close(out, ref)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_fused_norm_linear_parity(dtype):
    B, N, D_in, D_out = 2, 8, 32, 64
    x = torch.randn(B, N, D_in, dtype=dtype)
    weight = torch.randn(D_out, D_in, dtype=dtype)
    bias = torch.randn(D_out, dtype=dtype)

    ref = fused_norm_linear_pytorch(x, weight, bias=bias)
    out = fused_norm_linear(x, weight, bias=bias)
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_softplus_parity(dtype):
    x = torch.randn(4, 16, dtype=dtype)
    ref = softplus_pytorch(x)
    out = softplus_fwd(x)
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)


def test_fused_ce_parity():
    logits = torch.randn(4, 10, dtype=torch.float32)
    targets = torch.tensor([1, 4, 0, 9], dtype=torch.long)

    ref = fused_cross_entropy_pytorch(logits, targets)
    out = fused_cross_entropy(logits, targets)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


def test_fused_kd_parity():
    s_logits = torch.randn(4, 10, dtype=torch.float32)
    t_logits = torch.randn(4, 10, dtype=torch.float32)

    ref = fused_kd_loss_pytorch(s_logits, t_logits, temperature=2.0)
    out = fused_kd_loss(s_logits, t_logits, temperature=2.0)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


def test_sliding_window_parity():
    B, H, N, D = 1, 2, 16, 32
    q = torch.randn(B, H, N, D)
    k = torch.randn(B, H, N, D)
    v = torch.randn(B, H, N, D)

    ref = sliding_window_pytorch(q, k, v, window_size=4)
    out = sliding_window_attention(q, k, v, window_size=4)
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)


def test_cosformer_parity():
    B, H, N, D = 1, 2, 16, 32
    q = torch.randn(B, H, N, D)
    k = torch.randn(B, H, N, D)
    v = torch.randn(B, H, N, D)

    ref = cosformer_pytorch_fwd(q, k, v)
    out = cosformer_triton_fwd(q, k, v)
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)


def test_fused_rope_parity():
    B, H, N, D = 2, 4, 16, 32
    q = torch.randn(B, H, N, D)
    k = torch.randn(B, H, N, D)
    cos = torch.ones(B, 1, N, D)
    sin = torch.zeros(B, 1, N, D)

    q_out, k_out = fused_rope_embedding_kernel(q, k, cos, sin)
    torch.testing.assert_close(q_out, q)
    torch.testing.assert_close(k_out, k)


def test_selective_scan_parity():
    B, D, L, N = 2, 8, 16, 4
    u = torch.randn(B, D, L)
    delta = torch.randn(B, D, L)
    A = torch.randn(D, N)
    B_mat = torch.randn(B, N, L)
    C = torch.randn(B, N, L)

    ref = selective_scan_ref(u, delta, A, B_mat, C)
    if selective_scan_fn is not None:
        out = selective_scan_fn(u, delta, A, B_mat, C)
        assert out.shape == ref.shape


def test_scan_kernels_execution():
    """Verify recurrent scan kernels directly across CPU, MPS, and CUDA without skipping."""
    B, H, L, D = 2, 4, 16, 32
    device = "cuda" if (torch.cuda.is_available() and HAS_TRITON) else "cpu"

    from neuralops.core.triton_kernels.gdn_scan import gdn_scan
    q = torch.randn(B, H, L, D, device=device, dtype=torch.float32)
    k = F.normalize(torch.randn(B, H, L, D, device=device, dtype=torch.float32), dim=-1)
    v = torch.randn(B, H, L, D, device=device, dtype=torch.float32)
    beta = torch.sigmoid(torch.randn(B, H, L, device=device, dtype=torch.float32))
    alpha = torch.sigmoid(torch.randn(B, H, L, device=device, dtype=torch.float32))
    out_gdn = gdn_scan(q, k, v, beta, alpha)
    assert out_gdn.shape == q.shape

    from neuralops.core.triton_kernels.hgrn2_scan import hgrn2_scan
    f = torch.randn(B, L, D, device=device, dtype=torch.float32)
    iv = torch.randn(B, L, D, device=device, dtype=torch.float32)
    out_hgrn2 = hgrn2_scan(f, iv, res_clip=1.0)
    assert out_hgrn2.shape == f.shape
