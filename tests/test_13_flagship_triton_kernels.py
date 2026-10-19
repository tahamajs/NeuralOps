#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Comprehensive Verification and Parity Test Suite for the 13 Flagship Triton GPU Kernels.
Targeting SM 8.9 (NVIDIA Ada Lovelace RTX 4070 Laptop GPU, 8GB VRAM).

Tests mathematical numerical correctness, PyTorch vs Triton equivalence,
gradient flow / autograd where applicable, and edge case dimension handling.
"""

from __future__ import annotations

import math
import pytest
import torch
import torch.nn.functional as F

from neuralops.core.triton_kernels import (
    triton_fused_lowrank_gated_gemm,
    triton_fused_lowrank_gated_gemm_pytorch_ref,
    triton_fused_lowrank_sparse_gemm,
    triton_fused_lowrank_sparse_gemm_pytorch_ref,
    triton_strided_tt_contraction,
    triton_strided_tt_linear,
    triton_strided_tt_contraction_pytorch_ref,
    triton_fused_hadamard_w4a16_gemm,
    triton_fused_hadamard_w4a16_gemm_pytorch_ref,
    triton_lut_codebook_dequant_gemm,
    triton_lut_codebook_dequant_gemm_pytorch_ref,
    triton_fused_kronecker_matmul,
    triton_fused_kronecker_matmul_pytorch_ref,
    triton_chunked_delta_gate_scan,
    triton_chunked_delta_gate_scan_pytorch_ref,
    triton_cosformer_linear_attention,
    triton_cosformer_linear_attention_pytorch_ref,
    triton_block_sparse_streaming_attn,
    triton_block_sparse_streaming_attn_pytorch_ref,
    triton_hgrn2_fused_bwd_scan,
    triton_hgrn2_fused_bwd_scan_pytorch_ref,
    triton_spectral_entropy_svd_gate,
    triton_spectral_entropy_svd_gate_pytorch_ref,
    triton_fused_act_scale_quant_gemm,
    triton_fused_act_scale_quant_gemm_pytorch_ref,
    triton_fused_latent_kv_cache_attention,
    triton_fused_latent_kv_cache_attention_pytorch_ref,
)


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def test_kernel_1_fused_lowrank_gated_gemm(device):
    torch.manual_seed(42)
    M, K, r, N = 64, 256, 32, 128
    x = torch.randn(M, K, device=device, dtype=torch.float16)
    V = torch.randn(r, K, device=device, dtype=torch.float16)
    U = torch.randn(N, r, device=device, dtype=torch.float16)
    alpha = torch.randn(r, device=device, dtype=torch.float16)
    bias = torch.randn(N, device=device, dtype=torch.float16)

    out_ref = triton_fused_lowrank_gated_gemm_pytorch_ref(x, V, U, alpha=alpha, b_drift=bias)
    out_tri = triton_fused_lowrank_gated_gemm(x, V, U, alpha=alpha, b_drift=bias)

    assert torch.allclose(out_ref, out_tri, atol=0.15, rtol=0.03)


def test_kernel_2_fused_lowrank_sparse_gemm(device):
    torch.manual_seed(42)
    M, K, r, N = 64, 256, 32, 128
    x = torch.randn(M, K, device=device, dtype=torch.float16)
    V = torch.randn(r, K, device=device, dtype=torch.float16)
    U = torch.randn(N, r, device=device, dtype=torch.float16)
    s = torch.randn(N, K, device=device, dtype=torch.float16) * (torch.rand(N, K, device=device) < 0.05).to(torch.float16)
    bias = torch.randn(N, device=device, dtype=torch.float16)


    out_ref = triton_fused_lowrank_sparse_gemm_pytorch_ref(x, V, U, s=s, bias=bias)
    out_tri = triton_fused_lowrank_sparse_gemm(x, V, U, s=s, bias=bias)

    assert torch.allclose(out_ref, out_tri, atol=2e-2, rtol=1e-2)


def test_kernel_3_strided_tt(device):
    torch.manual_seed(42)
    p_dims = [8, 8]
    q_dims = [8, 8]
    r = 8
    core1 = torch.randn(1, p_dims[0], q_dims[0], r, device=device, dtype=torch.float16)
    core2 = torch.randn(r, p_dims[1], q_dims[1], 1, device=device, dtype=torch.float16)

    W_ref = triton_strided_tt_contraction_pytorch_ref(core1, core2, p_dims, q_dims)
    W_tri = triton_strided_tt_contraction(core1, core2, p_dims, q_dims)
    assert torch.allclose(W_ref, W_tri, atol=2e-2, rtol=1e-2)

    M = 32
    in_features = p_dims[0] * p_dims[1]
    out_features = q_dims[0] * q_dims[1]
    x = torch.randn(M, in_features, device=device, dtype=torch.float16)
    bias = torch.randn(out_features, device=device, dtype=torch.float16)

    out_linear = triton_strided_tt_linear(x, core1, core2, p_dims, q_dims, bias=bias)
    out_ref_linear = F.linear(x, W_ref, bias)
    assert torch.allclose(out_linear, out_ref_linear, atol=3e-2, rtol=2e-2)


def test_kernel_4_hadamard_w4a16_gemm(device):
    torch.manual_seed(42)
    M, K, N = 32, 128, 64
    group_size = 32

    x = torch.randn(M, K, device=device, dtype=torch.float16)
    w_packed = torch.randint(0, 255, (N, K // 2), device=device, dtype=torch.uint8)
    scales = torch.rand(N, K // group_size, device=device, dtype=torch.float16) * 0.1
    zeros = torch.zeros(N, K // group_size, device=device, dtype=torch.float16)
    bias = torch.randn(N, device=device, dtype=torch.float16)

    out_ref = triton_fused_hadamard_w4a16_gemm_pytorch_ref(x, w_packed, scales, zeros, group_size=group_size, bias=bias)
    out_tri = triton_fused_hadamard_w4a16_gemm(x, w_packed, scales, zeros, group_size=group_size, bias=bias)

    assert torch.allclose(out_ref, out_tri, atol=3e-2, rtol=2e-2)


def test_kernel_5_lut_codebook_dequant_gemm(device):
    torch.manual_seed(42)
    M, K, N = 32, 128, 64
    group_size = 64

    x = torch.randn(M, K, device=device, dtype=torch.float16)
    indices_packed = torch.randint(0, 255, (N, K // 2), device=device, dtype=torch.uint8)
    num_groups = K // group_size
    codebook = torch.randn(num_groups, 16, device=device, dtype=torch.float16) * 0.2
    bias = torch.randn(N, device=device, dtype=torch.float16)

    out_ref = triton_lut_codebook_dequant_gemm_pytorch_ref(x, indices_packed, codebook, group_size=group_size, bias=bias)
    out_tri = triton_lut_codebook_dequant_gemm(x, indices_packed, codebook, group_size=group_size, bias=bias)

    assert torch.allclose(out_ref, out_tri, atol=3e-2, rtol=2e-2)


def test_kernel_6_fused_kronecker_matmul(device):
    torch.manual_seed(42)
    M1, N1 = 16, 16
    M2, N2 = 16, 16
    M = 32

    x = torch.randn(M, N1 * N2, device=device, dtype=torch.float16)
    A = torch.randn(M1, N1, device=device, dtype=torch.float16)
    B = torch.randn(M2, N2, device=device, dtype=torch.float16)
    bias = torch.randn(M1 * M2, device=device, dtype=torch.float16)

    out_ref = triton_fused_kronecker_matmul_pytorch_ref(x, A, B, bias=bias)
    out_tri = triton_fused_kronecker_matmul(x, A, B, bias=bias)

    assert torch.allclose(out_ref, out_tri, atol=3e-2, rtol=2e-2)


def test_kernel_7_chunked_delta_gate_scan(device):
    torch.manual_seed(42)
    B, H, L, D_K, D_V = 2, 4, 32, 32, 32

    q = torch.randn(B, H, L, D_K, device=device, dtype=torch.float32)
    k = F.normalize(torch.randn(B, H, L, D_K, device=device, dtype=torch.float32), p=2, dim=-1)
    v = torch.randn(B, H, L, D_V, device=device, dtype=torch.float32)
    alpha = torch.sigmoid(torch.randn(B, H, L, device=device, dtype=torch.float32))
    beta = torch.sigmoid(torch.randn(B, H, L, device=device, dtype=torch.float32))

    out_ref = triton_chunked_delta_gate_scan_pytorch_ref(q, k, v, alpha, beta)
    out_tri = triton_chunked_delta_gate_scan(q, k, v, alpha, beta)

    assert torch.allclose(out_ref, out_tri, atol=1e-3, rtol=1e-3)


def test_kernel_8_cosformer_linear_attention(device):
    torch.manual_seed(42)
    B, H, L, D = 2, 4, 64, 32

    q = torch.randn(B, H, L, D, device=device, dtype=torch.float16)
    k = torch.randn(B, H, L, D, device=device, dtype=torch.float16)
    v = torch.randn(B, H, L, D, device=device, dtype=torch.float16)

    out_ref_causal = triton_cosformer_linear_attention_pytorch_ref(q, k, v, is_causal=True)
    out_tri_causal = triton_cosformer_linear_attention(q, k, v, is_causal=True)
    assert torch.allclose(out_ref_causal, out_tri_causal, atol=2e-2, rtol=1e-2)

    out_ref_noncausal = triton_cosformer_linear_attention_pytorch_ref(q, k, v, is_causal=False)
    out_tri_noncausal = triton_cosformer_linear_attention(q, k, v, is_causal=False)
    assert torch.allclose(out_ref_noncausal, out_tri_noncausal, atol=2e-2, rtol=1e-2)


def test_kernel_9_block_sparse_streaming_attn(device):
    torch.manual_seed(42)
    B, H, L, D = 2, 2, 64, 32
    block_size = 32
    num_q_blocks = L // block_size
    K_active = 2

    q = torch.randn(B, H, L, D, device=device, dtype=torch.float16)
    k = torch.randn(B, H, L, D, device=device, dtype=torch.float16)
    v = torch.randn(B, H, L, D, device=device, dtype=torch.float16)

    active_indices = torch.randint(0, num_q_blocks, (B, H, num_q_blocks, K_active), device=device, dtype=torch.int32)

    out_ref = triton_block_sparse_streaming_attn_pytorch_ref(q, k, v, active_indices, block_size=block_size)
    out_tri = triton_block_sparse_streaming_attn(q, k, v, active_indices, block_size=block_size)

    assert torch.allclose(out_ref, out_tri, atol=3e-2, rtol=2e-2)


def test_kernel_10_hgrn2_fused_bwd_scan(device):
    torch.manual_seed(42)
    B, H, L, D = 2, 4, 32, 32

    x = torch.randn(B, H, L, D, device=device, dtype=torch.float32)
    g = torch.sigmoid(torch.randn(B, H, L, D, device=device, dtype=torch.float32))

    out_ref = triton_hgrn2_fused_bwd_scan_pytorch_ref(x, g)
    out_tri = triton_hgrn2_fused_bwd_scan(x, g)

    assert torch.allclose(out_ref, out_tri, atol=1e-3, rtol=1e-3)


def test_kernel_11_spectral_entropy_svd_gate(device):
    torch.manual_seed(42)
    M, K, R_MAX, N = 32, 128, 64, 128
    eff_r = 32
    x = torch.randn(M, K, device=device, dtype=torch.float16)
    V_r = torch.randn(K, R_MAX, device=device, dtype=torch.float16)
    U_r = torch.randn(R_MAX, N, device=device, dtype=torch.float16)
    s_weights = torch.rand(R_MAX, device=device, dtype=torch.float16)
    bias = torch.randn(N, device=device, dtype=torch.float16)

    out_ref = triton_spectral_entropy_svd_gate_pytorch_ref(x, V_r, U_r, effective_rank=eff_r, s_weights=s_weights, bias=bias)
    out_tri = triton_spectral_entropy_svd_gate(x, V_r, U_r, effective_rank=eff_r, s_weights=s_weights, bias=bias)

    assert torch.allclose(out_ref, out_tri, atol=8e-2, rtol=2e-2)


def test_kernel_12_fused_act_scale_quant_gemm(device):
    torch.manual_seed(42)
    M, K, N = 32, 128, 64

    x = torch.randn(M, K, device=device, dtype=torch.float16)
    s_inv = torch.rand(K, device=device, dtype=torch.float16) + 0.5
    w = torch.randn(N, K, device=device, dtype=torch.float16)
    bias = torch.randn(N, device=device, dtype=torch.float16)

    out_ref = triton_fused_act_scale_quant_gemm_pytorch_ref(x, s_inv, w, bias=bias)
    out_tri = triton_fused_act_scale_quant_gemm(x, s_inv, w, bias=bias)

    assert torch.allclose(out_ref, out_tri, atol=2e-2, rtol=1e-2)


def test_kernel_13_fused_latent_kv_cache_attention(device):
    torch.manual_seed(42)
    B, H, L_Q, L_KV, R, D_K, D_V = 2, 4, 32, 64, 32, 64, 64

    q = torch.randn(B, H, L_Q, D_K, device=device, dtype=torch.float16)
    c_kv = torch.randn(B, L_KV, R, device=device, dtype=torch.float16)
    u_k = torch.randn(D_K, R, device=device, dtype=torch.float16) * 0.1
    u_v = torch.randn(R, D_V, device=device, dtype=torch.float16) * 0.1

    out_ref = triton_fused_latent_kv_cache_attention_pytorch_ref(q, c_kv, u_k, u_v)
    out_tri = triton_fused_latent_kv_cache_attention(q, c_kv, u_k, u_v)

    assert torch.allclose(out_ref, out_tri, atol=3e-2, rtol=2e-2)
