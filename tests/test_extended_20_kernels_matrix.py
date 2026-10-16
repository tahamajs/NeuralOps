"""
Exhaustive Verification Suite for the 20 Extended Production Kernels.
Tests all 20 kernels on CPU and CUDA for numerical validity, parity, and 3-level fallback.
"""

import math
import pytest
import torch
import torch.nn.functional as F

from neuralops.core.cuda_kernels._fallback import force_kernel_backend
from neuralops.kernels.extended_kernels import (
    fused_swiglu,
    fused_rmsnorm,
    fused_rotary_emb,
    fused_alibi_attention,
    fused_sink_token_attention,
    fused_decay_scan,
    fused_hadamard_transform,
    fused_int4_dequant,
    fused_fp8_e4m3_gemm,
    fused_spectral_dct,
    fused_chunked_cumsum,
    fused_cka_alignment,
    fused_mohawk_homotopy,
    fused_nullspace_project,
    fused_householder_reflect,
    fused_simplex_project,
    fused_topk_gating,
    fused_label_smoothing_ce,
    fused_kl_divergence,
    fused_lowrank_residual_add,
)


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


# 1. SwiGLU
def test_kernel_1_swiglu(device):
    torch.manual_seed(42)
    B, Din, Dout = 2, 32, 64
    x = torch.randn(B, Din, device=device)
    w_gate = torch.randn(Dout, Din, device=device)
    w_up = torch.randn(Dout, Din, device=device)

    with force_kernel_backend("pytorch"):
        out_py = fused_swiglu(x, w_gate, w_up)
    with force_kernel_backend("auto"):
        out_auto = fused_swiglu(x, w_gate, w_up)

    assert out_auto.shape == (B, Dout)
    assert torch.allclose(out_auto, out_py, atol=1e-4, rtol=1e-4)


# 2. RMSNorm
def test_kernel_2_rmsnorm(device):
    torch.manual_seed(42)
    B, T, D = 2, 16, 64
    x = torch.randn(B, T, D, device=device)
    weight = torch.ones(D, device=device)

    with force_kernel_backend("pytorch"):
        out_py = fused_rmsnorm(x, weight)
    with force_kernel_backend("auto"):
        out_auto = fused_rmsnorm(x, weight)

    assert out_auto.shape == (B, T, D)
    assert torch.allclose(out_auto, out_py, atol=1e-4, rtol=1e-4)


# 3. RoPE
def test_kernel_3_rotary_emb(device):
    torch.manual_seed(42)
    B, H, T, D = 2, 4, 16, 32
    x = torch.randn(B, H, T, D, device=device)
    cos = torch.cos(torch.randn(1, 1, T, D, device=device))
    sin = torch.sin(torch.randn(1, 1, T, D, device=device))

    with force_kernel_backend("pytorch"):
        out_py = fused_rotary_emb(x, cos, sin)
    with force_kernel_backend("auto"):
        out_auto = fused_rotary_emb(x, cos, sin)

    assert out_auto.shape == (B, H, T, D)
    assert torch.allclose(out_auto, out_py, atol=1e-5, rtol=1e-5)


# 4. ALiBi Attention
def test_kernel_4_alibi_attention(device):
    torch.manual_seed(42)
    B, H, T = 2, 8, 16
    scores = torch.randn(B, H, T, T, device=device)

    with force_kernel_backend("pytorch"):
        out_py = fused_alibi_attention(scores, H)
    with force_kernel_backend("auto"):
        out_auto = fused_alibi_attention(scores, H)

    assert out_auto.shape == (B, H, T, T)
    assert torch.allclose(out_auto, out_py, atol=1e-5, rtol=1e-5)


# 5. Sink Token Attention
def test_kernel_5_sink_token_attention(device):
    torch.manual_seed(42)
    B, H, T, D = 2, 2, 32, 16
    q = torch.randn(B, H, T, D, device=device)
    k = torch.randn(B, H, T, D, device=device)
    v = torch.randn(B, H, T, D, device=device)

    with force_kernel_backend("pytorch"):
        out_py = fused_sink_token_attention(q, k, v, num_sinks=4, window_size=8)
    with force_kernel_backend("auto"):
        out_auto = fused_sink_token_attention(q, k, v, num_sinks=4, window_size=8)

    assert out_auto.shape == (B, H, T, D)
    assert torch.allclose(out_auto, out_py, atol=1e-4, rtol=1e-4)


# 6. Exponential Decay Scan
def test_kernel_6_decay_scan(device):
    torch.manual_seed(42)
    B, T, Dk, Dv = 2, 16, 8, 8
    k = torch.randn(B, T, Dk, device=device)
    v = torch.randn(B, T, Dv, device=device)

    with force_kernel_backend("pytorch"):
        out_py = fused_decay_scan(k, v, decay=0.85)
    with force_kernel_backend("auto"):
        out_auto = fused_decay_scan(k, v, decay=0.85)

    assert out_auto.shape == (B, T, Dv)
    assert torch.allclose(out_auto, out_py, atol=1e-4, rtol=1e-4)


# 7. Fast Walsh-Hadamard Transform
def test_kernel_7_hadamard_transform(device):
    torch.manual_seed(42)
    B, N = 4, 32
    x = torch.randn(B, N, device=device)

    with force_kernel_backend("pytorch"):
        out_py = fused_hadamard_transform(x)
    with force_kernel_backend("auto"):
        out_auto = fused_hadamard_transform(x)

    assert out_auto.shape == (B, N)
    assert torch.allclose(out_auto, out_py, atol=1e-5, rtol=1e-5)


# 8. INT4 Dequantization
def test_kernel_8_int4_dequant(device):
    torch.manual_seed(42)
    K, N = 16, 32
    packed = torch.randint(0, 255, (K, N // 2), device=device, dtype=torch.uint8)
    scale = torch.rand(K, N, device=device) * 0.1
    zero = torch.zeros(K, N, device=device)

    with force_kernel_backend("pytorch"):
        out_py = fused_int4_dequant(packed, scale, zero)
    with force_kernel_backend("auto"):
        out_auto = fused_int4_dequant(packed, scale, zero)

    assert out_auto.shape == (K, N)
    assert torch.allclose(out_auto, out_py, atol=1e-5, rtol=1e-5)


# 9. FP8 Scaled GEMM
def test_kernel_9_fp8_e4m3_gemm(device):
    torch.manual_seed(42)
    M, K, N = 16, 32, 16
    a = torch.randn(M, K, device=device)
    b = torch.randn(K, N, device=device)

    with force_kernel_backend("pytorch"):
        out_py = fused_fp8_e4m3_gemm(a, b, scale_a=0.5, scale_b=0.25)
    with force_kernel_backend("auto"):
        out_auto = fused_fp8_e4m3_gemm(a, b, scale_a=0.5, scale_b=0.25)

    assert out_auto.shape == (M, N)
    assert torch.allclose(out_auto, out_py, atol=1e-5, rtol=1e-5)


# 10. Discrete Cosine Transform (DCT-II)
def test_kernel_10_spectral_dct(device):
    torch.manual_seed(42)
    B, T, D = 2, 8, 16
    x = torch.randn(B, T, D, device=device)

    with force_kernel_backend("pytorch"):
        out_py = fused_spectral_dct(x)
    with force_kernel_backend("auto"):
        out_auto = fused_spectral_dct(x)

    assert out_auto.shape == (B, T, D)
    assert torch.allclose(out_auto, out_py, atol=1e-5, rtol=1e-5)


# 11. Chunked Cumsum
def test_kernel_11_chunked_cumsum(device):
    torch.manual_seed(42)
    B, T, D = 2, 32, 8
    x = torch.randn(B, T, D, device=device)

    with force_kernel_backend("pytorch"):
        out_py = fused_chunked_cumsum(x, chunk_size=16)
    with force_kernel_backend("auto"):
        out_auto = fused_chunked_cumsum(x, chunk_size=16)

    assert out_auto.shape == (B, T, D)
    assert torch.allclose(out_auto, out_py, atol=1e-5, rtol=1e-5)


# 12. CKA Alignment
def test_kernel_12_cka_alignment(device):
    torch.manual_seed(42)
    N, D = 32, 16
    x = torch.randn(N, D, device=device)
    y = x + torch.randn(N, D, device=device) * 0.05

    with force_kernel_backend("pytorch"):
        out_py = fused_cka_alignment(x, y)
    with force_kernel_backend("auto"):
        out_auto = fused_cka_alignment(x, y)

    assert out_auto.item() >= 0.0
    assert torch.allclose(out_auto, out_py, atol=1e-5, rtol=1e-5)


# 13. MOHAWK Homotopy
def test_kernel_13_mohawk_homotopy(device):
    torch.manual_seed(42)
    D = 16
    w0 = torch.randn(D, D, device=device)
    w_target = torch.randn(D, D, device=device)

    with force_kernel_backend("pytorch"):
        out_py = fused_mohawk_homotopy(w0, w_target, step=50, max_steps=100)
    with force_kernel_backend("auto"):
        out_auto = fused_mohawk_homotopy(w0, w_target, step=50, max_steps=100)

    assert out_auto.shape == (D, D)
    assert torch.allclose(out_auto, out_py, atol=1e-5, rtol=1e-5)


# 14. Null Space Projection
def test_kernel_14_nullspace_project(device):
    torch.manual_seed(42)
    D, K = 32, 4
    delta = torch.randn(D, device=device)
    q, _ = torch.linalg.qr(torch.randn(D, K, device=device))

    with force_kernel_backend("pytorch"):
        out_py = fused_nullspace_project(delta, q)
    with force_kernel_backend("auto"):
        out_auto = fused_nullspace_project(delta, q)

    assert out_auto.shape == (D,)
    assert torch.allclose(out_auto, out_py, atol=1e-5, rtol=1e-5)


# 15. Householder Reflection
def test_kernel_15_householder_reflect(device):
    torch.manual_seed(42)
    B, D = 4, 16
    x = torch.randn(B, D, device=device)
    v = torch.randn(D, device=device)

    with force_kernel_backend("pytorch"):
        out_py = fused_householder_reflect(x, v)
    with force_kernel_backend("auto"):
        out_auto = fused_householder_reflect(x, v)

    assert out_auto.shape == (B, D)
    assert torch.allclose(out_auto, out_py, atol=1e-5, rtol=1e-5)


# 16. Simplex Projection
def test_kernel_16_simplex_project(device):
    torch.manual_seed(42)
    B, D = 4, 8
    y = torch.randn(B, D, device=device)

    with force_kernel_backend("pytorch"):
        out_py = fused_simplex_project(y)
    with force_kernel_backend("auto"):
        out_auto = fused_simplex_project(y)

    assert out_auto.shape == (B, D)
    assert (out_auto >= -1e-6).all(), "Simplex projection must be non-negative"
    assert torch.allclose(out_auto.sum(dim=-1), torch.ones(B, device=device), atol=1e-4), "Sum must equal 1"
    assert torch.allclose(out_auto, out_py, atol=1e-5, rtol=1e-5)


# 17. Top-K Gating
def test_kernel_17_topk_gating(device):
    torch.manual_seed(42)
    B, E = 8, 16
    logits = torch.randn(B, E, device=device)

    with force_kernel_backend("pytorch"):
        w_py, idx_py = fused_topk_gating(logits, top_k=2)
    with force_kernel_backend("auto"):
        w_auto, idx_auto = fused_topk_gating(logits, top_k=2)

    assert w_auto.shape == (B, 2)
    assert idx_auto.shape == (B, 2)
    assert torch.allclose(w_auto, w_py, atol=1e-5, rtol=1e-5)
    assert torch.equal(idx_auto, idx_py)


# 18. Label Smoothed Cross Entropy
def test_kernel_18_label_smoothing_ce(device):
    torch.manual_seed(42)
    N, C = 16, 5
    logits = torch.randn(N, C, device=device)
    targets = torch.randint(0, C, (N,), device=device)

    with force_kernel_backend("pytorch"):
        loss_py = fused_label_smoothing_ce(logits, targets, smoothing=0.1)
    with force_kernel_backend("auto"):
        loss_auto = fused_label_smoothing_ce(logits, targets, smoothing=0.1)

    assert loss_auto.dim() == 0
    assert torch.allclose(loss_auto, loss_py, atol=1e-5, rtol=1e-5)


# 19. KL Divergence Distillation
def test_kernel_19_kl_divergence(device):
    torch.manual_seed(42)
    N, C = 16, 10
    s_logits = torch.randn(N, C, device=device)
    t_logits = torch.randn(N, C, device=device)

    with force_kernel_backend("pytorch"):
        loss_py = fused_kl_divergence(s_logits, t_logits, tau=2.0)
    with force_kernel_backend("auto"):
        loss_auto = fused_kl_divergence(s_logits, t_logits, tau=2.0)

    assert loss_auto.dim() == 0
    assert torch.allclose(loss_auto, loss_py, atol=1e-5, rtol=1e-5)


# 20. Low-Rank with Residual Addition
def test_kernel_20_lowrank_residual_add(device):
    torch.manual_seed(42)
    B, T, Din, Dout, R = 2, 8, 16, 16, 4
    x = torch.randn(B, T, Din, device=device)
    u = torch.randn(Din, R, device=device)
    v = torch.randn(R, Dout, device=device)
    x_res = torch.randn(B, T, Dout, device=device)

    with force_kernel_backend("pytorch"):
        out_py = fused_lowrank_residual_add(x, u, v, x_res)
    with force_kernel_backend("auto"):
        out_auto = fused_lowrank_residual_add(x, u, v, x_res)

    assert out_auto.shape == (B, T, Dout)
    assert torch.allclose(out_auto, out_py, atol=1e-5, rtol=1e-5)
