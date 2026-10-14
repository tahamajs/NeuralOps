"""Hardware and Numerical Tests for 4 New Architecture Capabilities and FP8 Low-Rank MMA.

Tests:
1. Native Ada Lovelace FP8 (torch.float8_e4m3fn) Tensor Core Matmul & Quantization
2. SpectraFormer Fused Complex Spectral Gate + RMS Variance Matching
3. KDA / HLA Fused Triton Chunk Scan
4. NEXUS / Orion Fused Top-k Sparse Route & Combine
"""

import pytest
import torch
import math

from neuralops.core.triton_kernels.low_rank import (
    quantize_to_fp8_dynamic,
    fp8_tensorcore_matmul_triton,
)
from neuralops.core.triton_kernels.elementwise import (
    fused_spectral_gate_triton,
    fused_rms_variance_match_triton,
)
from neuralops.core.triton_kernels.gla_scan import (
    kda_hla_chunk_scan_triton,
)
from neuralops.core.triton_kernels.sparse_attn import (
    fused_topk_sparse_route_and_combine_triton,
)


def test_fp8_quantization_and_matmul():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    M, K, N = 16, 64, 32
    x = torch.randn(M, K, device=device, dtype=torch.float32)
    b = torch.randn(K, N, device=device, dtype=torch.float32)

    x_fp8, s_x = quantize_to_fp8_dynamic(x)
    b_fp8, s_b = quantize_to_fp8_dynamic(b)

    assert x_fp8.shape == x.shape
    assert b_fp8.shape == b.shape
    assert (s_x > 0).all()
    assert (s_b > 0).all()

    # Execute fused FP8 Tensor Core matmul or fallback
    y = fp8_tensorcore_matmul_triton(x_fp8, b_fp8, s_x, s_b, out_dtype=torch.float32)
    assert y.shape == (M, N)
    assert not torch.isnan(y).any()
    assert not torch.isinf(y).any()

    # Compare with reference high precision
    y_ref = torch.matmul(x, b)
    cos_sim = torch.cosine_similarity(y.flatten(), y_ref.flatten(), dim=0).item()
    print(f"FP8 matmul cos_sim: {cos_sim:.4f}")
    assert cos_sim > 0.85, f"FP8 matmul cos_sim too low: {cos_sim:.4f}"


def test_spectraformer_spectral_gate_and_rms():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, H, N, D = 2, 2, 8, 32
    # Complex-valued tensors
    x_re = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
    x_im = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
    x = torch.complex(x_re, x_im)

    w_re = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
    w_im = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
    w = torch.complex(w_re, w_im)
    tau = torch.tensor(0.01, device=device, dtype=torch.float32)

    # 1. Complex spectral gate
    out_gate = fused_spectral_gate_triton(x, w, tau)
    assert out_gate.shape == x.shape
    assert not torch.isnan(out_gate).any()

    # 2. Fused RMS variance match
    ctx = torch.randn(B, H, N, D, device=device, dtype=torch.float32) * 2.0
    v = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
    matched = fused_rms_variance_match_triton(ctx, v, clamp_min=0.25, clamp_max=4.0)
    assert matched.shape == ctx.shape
    assert not torch.isnan(matched).any()


def test_kda_hla_chunk_scan():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, H, T, D = 2, 2, 16, 32
    q = torch.randn(B, H, T, D, device=device, dtype=torch.float32)
    k = torch.randn(B, H, T, D, device=device, dtype=torch.float32)
    v = torch.randn(B, H, T, D, device=device, dtype=torch.float32)
    log_decay = -torch.abs(torch.randn(B, H, T, 1, device=device, dtype=torch.float32))

    y = kda_hla_chunk_scan_triton(q, k, v, log_decay, scale=1.0)
    assert y.shape == (B, H, T, D) or y.shape == (B, T, D)
    assert not torch.isnan(y).any()


def test_nexus_orion_topk_sparse_route():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, T, D = 2, 8, 32
    b1 = torch.randn(B, T, D, device=device, dtype=torch.float32)
    b2 = torch.randn(B, T, D, device=device, dtype=torch.float32)
    router_weights = torch.softmax(torch.randn(B, T, 2, device=device, dtype=torch.float32), dim=-1)

    y = fused_topk_sparse_route_and_combine_triton([b1, b2], router_weights)
    assert y.shape == b1.shape
    assert not torch.isnan(y).any()
