# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Sampled Hardware Kernels & Fallback Verification Suite (Wave 1 Audit).

Tests 20 representative CUDA/Triton kernels from neuralops.ops:
- fast_gelu, swiglu, rmsnorm, fused_cross_entropy, fused_dropout_add
- fused_norm_linear, fused_rotary_emb, kronecker_matmul, lut_codebook_dequant_gemm
- linear_attention, fused_lowrank_gated_gemm, chunked_delta_gate_scan, hgrn2_fused_bwd_scan
- cosformer_linear_attention, rpca_sparse_svd_gemm, moe_routing, selective_scan
- differential_attention, alibi_attention, act_scale_gemm

Verifies:
1. Clean importability without hardware errors.
2. Graceful Level 3 (PyTorch reference) fallback when CUDA is not present.
3. Correct tensor output shapes and numerical stability (no NaNs/Infs).
4. Triton kernel module importability and CPU fallback dispatch.
"""

from __future__ import annotations

import importlib
from typing import Any

import pytest
import torch

import neuralops.ops.cuda as noc
from neuralops.ops.cuda._fallback import force_kernel_backend


SAMPLED_CUDA_KERNEL_NAMES = [
    "fast_gelu",
    "swiglu",
    "rmsnorm",
    "fused_cross_entropy",
    "fused_dropout_add",
    "fused_norm_linear",
    "fused_rotary_emb",
    "kronecker_matmul",
    "lut_codebook_dequant_gemm",
    "linear_attention",
    "fused_lowrank_gated_gemm",
    "chunked_delta_gate_scan",
    "hgrn2_fused_bwd_scan",
    "cosformer_linear_attention",
    "rpca_sparse_svd_gemm",
    "moe_routing",
    "selective_scan",
    "differential_attention",
    "alibi_attention",
    "act_scale_gemm",
]

SAMPLED_TRITON_MODULES = [
    "neuralops.ops.triton.adaptive_sparse_triton",
    "neuralops.ops.triton.block_sparse",
    "neuralops.ops.triton.cosformer_triton_kernel",
    "neuralops.ops.triton.cur_linear",
    "neuralops.ops.triton.delta_net_scan",
    "neuralops.ops.triton.differential_attention",
    "neuralops.ops.triton.elementwise",
    "neuralops.ops.triton.fused_ce",
    "neuralops.ops.triton.fused_kd",
    "neuralops.ops.triton.fused_norm_linear",
    "neuralops.ops.triton.fused_rope",
    "neuralops.ops.triton.fused_svd_projection_triton",
    "neuralops.ops.triton.gdn_scan",
    "neuralops.ops.triton.gla_scan",
    "neuralops.ops.triton.gqa",
    "neuralops.ops.triton.hgrn2_scan",
    "neuralops.ops.triton.lightning_scan",
    "neuralops.ops.triton.linear_attn",
    "neuralops.ops.triton.low_rank",
    "neuralops.ops.triton.selective_scan",
]


@pytest.mark.parametrize("kernel_name", SAMPLED_CUDA_KERNEL_NAMES)
def test_operator_symbol_importability(kernel_name: str) -> None:
    """Verify that each sampled kernel operator can be imported and is callable."""
    op = getattr(noc, kernel_name, None)
    assert op is not None, f"Kernel {kernel_name} could not be resolved from neuralops.ops.cuda"
    assert callable(op), f"Kernel {kernel_name} is not callable"


def test_operator_modules_importability() -> None:
    """Verify that 20 sampled operator modules import cleanly even when GPU acceleration is absent."""
    for mod_path in SAMPLED_TRITON_MODULES:
        mod = importlib.import_module(mod_path)
        assert mod is not None, f"Module {mod_path} failed to import"


def test_fallback_activation_and_norm_kernels() -> None:
    """Test fallback execution for activation and normalization kernels."""
    # fast_gelu
    x = torch.randn(2, 16)
    out = noc.fast_gelu(x)
    assert out.shape == (2, 16)
    assert not torch.isnan(out).any()

    # swiglu
    x = torch.randn(2, 32)
    out = noc.swiglu(x)
    assert out.shape == (2, 16)
    assert not torch.isnan(out).any()

    # rmsnorm
    x = torch.randn(2, 16)
    w = torch.ones(16)
    out = noc.rmsnorm(x, w)
    assert out.shape == (2, 16)
    assert not torch.isnan(out).any()

    # fused_dropout_add
    x = torch.randn(2, 16)
    res = torch.randn(2, 16)
    out = noc.fused_dropout_add(x, res, 0.1, False)
    assert out.shape == (2, 16)
    assert not torch.isnan(out).any()

    # fused_cross_entropy
    logits = torch.randn(4, 10)
    targets = torch.tensor([1, 2, 0, 3])
    loss = noc.fused_cross_entropy(logits, targets)
    assert loss.dim() == 0
    assert not torch.isnan(loss)


def test_fallback_gemm_and_linear_kernels() -> None:
    """Test fallback execution for GEMM and linear projections."""
    # fused_norm_linear
    x = torch.randn(2, 16)
    w = torch.randn(32, 16)
    b = torch.randn(32)
    out = noc.fused_norm_linear(x, w, b)
    assert out.shape == (2, 32)
    assert not torch.isnan(out).any()

    # kronecker_matmul
    A = torch.randn(4, 4)
    B = torch.randn(8, 8)
    x = torch.randn(2, 32)
    out = noc.kronecker_matmul(x, A, B)
    assert out.shape == (2, 32)
    assert not torch.isnan(out).any()

    # lut_codebook_dequant_gemm
    x = torch.randn(2, 64)
    cb = torch.randn(16)
    indices = torch.randint(0, 255, (32, 32), dtype=torch.uint8)
    out = noc.lut_codebook_dequant_gemm(x, indices, cb, group_size=64)
    assert out.shape == (2, 32)
    assert not torch.isnan(out).any()

    # act_scale_gemm
    x = torch.randn(2, 16)
    s_inv = torch.ones(16)
    w = torch.randn(32, 16)
    out = noc.act_scale_gemm(x, s_inv, w)
    assert out.shape == (2, 32)
    assert not torch.isnan(out).any()

    # fused_lowrank_gated_gemm
    x = torch.randn(2, 32)
    V = torch.randn(8, 32)
    U = torch.randn(16, 8)
    out = noc.fused_lowrank_gated_gemm(x, V, U)
    assert out.shape == (2, 16)
    assert not torch.isnan(out).any()

    # rpca_sparse_svd_gemm
    x = torch.randn(2, 32)
    v_r = torch.randn(8, 32)
    u_r = torch.randn(16, 8)
    out = noc.rpca_sparse_svd_gemm(x, v_r, u_r)
    assert out.shape == (2, 16)
    assert not torch.isnan(out).any()


def test_fallback_attention_and_scan_kernels() -> None:
    """Test fallback execution for attention and scan mechanisms."""
    # fused_rotary_emb
    x = torch.randn(2, 4, 8, 16)
    cos = torch.randn(1, 1, 8, 16)
    sin = torch.randn(1, 1, 8, 16)
    out = noc.fused_rotary_emb(x, cos, sin)
    assert out.shape == (2, 4, 8, 16)
    assert not torch.isnan(out).any()

    # linear_attention
    q = torch.randn(2, 4, 8, 16)
    k = torch.randn(2, 4, 8, 16)
    v = torch.randn(2, 4, 8, 16)
    out = noc.linear_attention(q, k, v)
    assert out.shape == (2, 4, 8, 16)
    assert not torch.isnan(out).any()

    # cosformer_linear_attention
    out = noc.cosformer_linear_attention(q, k, v)
    assert out.shape == (2, 4, 8, 16)
    assert not torch.isnan(out).any()

    # differential_attention
    q1 = torch.randn(2, 4, 8, 16)
    k1 = torch.randn(2, 4, 8, 16)
    q2 = torch.randn(2, 4, 8, 16)
    k2 = torch.randn(2, 4, 8, 16)
    out = noc.differential_attention(q1, k1, q2, k2, v)
    assert out.shape == (2, 4, 8, 16)
    assert not torch.isnan(out).any()

    # alibi_attention
    slopes = torch.randn(4)
    out = noc.alibi_attention(q, k, v, slopes)
    assert out.shape == (2, 4, 8, 16)
    assert not torch.isnan(out).any()

    # chunked_delta_gate_scan
    B, H, S, D = 2, 2, 16, 16
    q = torch.randn(B, H, S, D)
    k = torch.randn(B, H, S, D)
    v = torch.randn(B, H, S, D)
    alpha = torch.randn(B, H, S, D)
    beta = torch.randn(B, H, S, D)
    out = noc.chunked_delta_gate_scan(q, k, v, alpha, beta)
    assert out.shape == (B, H, S, D)
    assert not torch.isnan(out).any()

    # hgrn2_fused_bwd_scan
    x_h = torch.randn(2, 2, 16, 8)
    g_h = torch.randn(2, 2, 16, 8)
    out = noc.hgrn2_fused_bwd_scan(x_h, g_h)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 2, 16, 8)
    assert not torch.isnan(out).any()

    # moe_routing
    tokens = torch.randn(4, 16)
    gate_logits = torch.randn(4, 8)
    out = noc.moe_routing(tokens, gate_logits)
    assert out.shape == (4, 16)
    assert not torch.isnan(out).any()

    # selective_scan
    u = torch.randn(2, 8, 16)
    delta = torch.randn(2, 8, 16)
    A = torch.randn(8, 4)
    B = torch.randn(2, 4, 16)
    C = torch.randn(2, 4, 16)
    y, s = noc.selective_scan(u, delta, A, B, C)
    assert y.shape == (2, 8, 16)
    assert not torch.isnan(y).any()


def test_force_kernel_backend_context() -> None:
    """Verify that force_kernel_backend properly selects PyTorch reference."""
    with force_kernel_backend("pytorch"):
        x = torch.randn(2, 16)
        out = noc.fast_gelu(x)
        assert out.shape == (2, 16)
        assert not torch.isnan(out).any()
