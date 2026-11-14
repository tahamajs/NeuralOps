#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Sample and Smoke Test Suite for NeuralOps Triton Hardware Operators.

Audits and verifies priority kernels and sampled Triton operations:
1. Module imports and symbol availability
2. Numerical validity (zero NaN/Inf) with fixed RNG seed
3. Parity against pure-PyTorch reference implementations
4. Autograd backward pass correctness
5. BLOCK_SIZE, SRAM estimation, and safe autotune configurations
6. Device fallback on non-CUDA environments (Rule 2.5)
7. Governance: strict prohibition of deprecated torch.cuda.amp (Rule 2.1)
"""

from __future__ import annotations

import ast
from pathlib import Path
import pytest
import torch
import torch.nn.functional as F

from neuralops.ops.triton._types import (
    BLOCK_SIZE,
    MAX_TRITON_SEQ_LEN,
    SUPPORTED_HEAD_DIMS,
)
from neuralops.ops.triton.autotuner import (
    estimate_kernel_sram_bytes,
    get_safe_autotune_configs,
)


@pytest.fixture(autouse=True)
def set_fixed_seed():
    """Set deterministic seed before each test."""
    torch.manual_seed(42)


def test_priority_imports_and_exports():
    """Verify all 12 priority kernel modules import cleanly and export canonical interfaces."""
    import neuralops.ops.triton.cross_entropy as ce
    import neuralops.ops.triton.flash_attention as fa
    import neuralops.ops.triton.fused_linear_cross_entropy as flce
    import neuralops.ops.triton.gdn as gdn
    import neuralops.ops.triton.gla as gla
    import neuralops.ops.triton.kldiv as kd
    import neuralops.ops.triton.layernorm as ln
    import neuralops.ops.triton.paged_attention as pa
    import neuralops.ops.triton.rmsnorm as rn
    import neuralops.ops.triton.rope as rp
    import neuralops.ops.triton.softmax as sm
    import neuralops.ops.triton.swiglu as sw

    assert callable(fa.flash_attention)
    assert callable(fa.flash_decoding_attention)
    assert callable(sm.softplus)
    assert callable(rn.fused_rmsnorm)
    assert callable(ln.layernorm_triton)
    assert callable(sw.fused_swiglu)
    assert callable(rp.fused_rope_embedding_kernel)
    assert callable(ce.fused_cross_entropy)
    assert callable(kd.fused_kd_loss)
    assert callable(pa.paged_attention_kernel)
    assert callable(gla.gla_scan)
    assert callable(gdn.gdn_scan_triton)
    assert callable(flce.fused_linear_cross_entropy)


def test_flash_attention_forward_and_ref_parity():
    """Verify Flash-Decoding attention executes, matches reference, and has no NaNs/Infs."""
    from neuralops.ops.triton.flash_attention import (
        flash_attention,
        flash_decoding_pytorch,
    )

    B, H, N_q, D = 2, 4, 1, 64
    N_kv = 48
    q = torch.randn(B, H, N_q, D)
    k = torch.randn(B, H, N_kv, D)
    v = torch.randn(B, H, N_kv, D)

    out = flash_attention(q, k, v, chunk_size=16)
    ref = flash_decoding_pytorch(q, k, v, chunk_size=16)

    assert not torch.isnan(out).any(), "Flash decoding output contains NaN"
    assert not torch.isinf(out).any(), "Flash decoding output contains Inf"
    diff = (out - ref).abs().max().item()
    assert diff < 1e-5, f"Flash decoding mismatch vs ref: {diff}"


def test_softmax_and_activations_parity():
    """Verify softmax/activation functions execute with valid numerical outputs."""
    from neuralops.ops.triton.softmax import (
        fast_gelu,
        fused_entmax_bisect_pytorch,
        softplus,
        softplus_triton,
    )

    x = torch.randn(4, 32)

    # softplus
    sp_out = softplus(x)
    sp_triton = softplus_triton(x)
    sp_ref = F.softplus(x)
    assert not torch.isnan(sp_out).any()
    assert (sp_out - sp_ref).abs().max().item() < 1e-6
    assert (sp_triton - sp_ref).abs().max().item() < 1e-6

    # fast gelu
    fg_out = fast_gelu(x)
    fg_ref = F.gelu(x, approximate="tanh")
    assert (fg_out - fg_ref).abs().max().item() < 1e-6

    # entmax
    ent_out = fused_entmax_bisect_pytorch(x, dim=-1)
    assert not torch.isnan(ent_out).any()
    # Entmax output sums to 1
    assert torch.allclose(ent_out.sum(dim=-1), torch.ones(4), atol=1e-4)


def test_rmsnorm_forward_and_ref_parity():
    """Verify RMSNorm forward matches PyTorch reference with zero NaNs/Infs."""
    from neuralops.ops.triton.rmsnorm import fused_rmsnorm, rmsnorm_triton

    x = torch.randn(4, 16, 64)
    w = torch.ones(64)

    out1 = fused_rmsnorm(x, w)
    out2 = rmsnorm_triton(x, w)
    ref = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)) * w

    assert not torch.isnan(out1).any()
    assert not torch.isinf(out1).any()
    assert (out1 - ref).abs().max().item() < 1e-6
    assert (out2 - ref).abs().max().item() < 1e-6


def test_layernorm_forward_and_ref_parity():
    """Verify LayerNorm forward with residual matches reference."""
    from neuralops.ops.triton.layernorm import layernorm_triton

    x = torch.randn(4, 16, 64)
    res = torch.randn(4, 16, 64)
    w = torch.ones(64)
    b = torch.zeros(64)

    out = layernorm_triton(x, w, bias=b, residual=res)
    ref = F.layer_norm(x + res, (64,), weight=w, bias=b, eps=1e-6)

    assert not torch.isnan(out).any()
    assert (out - ref).abs().max().item() < 1e-5


def test_swiglu_forward_and_ref_parity():
    """Verify SwiGLU forward matches SiLU(gate) * x."""
    from neuralops.ops.triton.swiglu import fused_swiglu

    x = torch.randn(4, 16, 64)
    gate = torch.randn(4, 16, 64)

    out = fused_swiglu(x, gate)
    ref = F.silu(gate) * x

    assert not torch.isnan(out).any()
    assert (out - ref).abs().max().item() < 1e-6


def test_rope_forward_and_backward():
    """Verify RoPE forward rotation and autograd backward gradient."""
    from neuralops.ops.triton.rope import (
        apply_rope_rotation,
        fused_rope_embedding_kernel,
    )

    B, H, N, D = 2, 4, 16, 64
    q = torch.randn(B, H, N, D, requires_grad=True)
    k = torch.randn(B, H, N, D, requires_grad=True)
    cos = torch.cos(torch.randn(1, 1, N, D // 2))
    sin = torch.sin(torch.randn(1, 1, N, D // 2))

    out_q, out_k = fused_rope_embedding_kernel(q, k, cos, sin)
    ref_q = apply_rope_rotation(q, cos, sin)
    ref_k = apply_rope_rotation(k, cos, sin)

    assert not torch.isnan(out_q).any() and not torch.isnan(out_k).any()
    assert (out_q - ref_q).abs().max().item() < 1e-6
    assert (out_k - ref_k).abs().max().item() < 1e-6

    loss = (out_q + out_k).sum()
    loss.backward()
    assert q.grad is not None and not torch.isnan(q.grad).any()
    assert k.grad is not None and not torch.isnan(k.grad).any()


def test_cross_entropy_forward_and_backward():
    """Verify FusedCrossEntropy matches PyTorch cross_entropy with backward."""
    from neuralops.ops.triton.cross_entropy import (
        fused_cross_entropy,
        fused_cross_entropy_pytorch,
    )

    logits = torch.randn(8, 50, requires_grad=True)
    targets = torch.randint(0, 50, (8,))

    loss = fused_cross_entropy(logits, targets)
    ref_loss = fused_cross_entropy_pytorch(logits, targets)

    assert not torch.isnan(loss).any()
    assert (loss - ref_loss).abs().item() < 1e-6

    loss.backward()
    assert logits.grad is not None and not torch.isnan(logits.grad).any()


def test_kldiv_forward_and_backward():
    """Verify FusedKDLoss matches PyTorch KD reference with backward."""
    from neuralops.ops.triton.kldiv import fused_kd_loss, fused_kd_loss_pytorch

    s_logits = torch.randn(4, 60, requires_grad=True)
    t_logits = torch.randn(4, 60)

    loss = fused_kd_loss(s_logits, t_logits, temperature=2.0)
    ref_loss = fused_kd_loss_pytorch(s_logits, t_logits, temperature=2.0)

    assert not torch.isnan(loss).any()
    assert (loss - ref_loss).abs().item() < 1e-6

    loss.backward()
    assert s_logits.grad is not None and not torch.isnan(s_logits.grad).any()


def test_paged_attention_decode_parity():
    """Verify PagedAttention decode kernel matches pure PyTorch reference."""
    from neuralops.ops.triton.paged_attention import (
        paged_attention_kernel,
        paged_attention_pytorch,
    )

    B, H, D = 2, 4, 32
    num_blocks, block_size = 8, 16
    q = torch.randn(B, H, D)
    k_cache = torch.randn(num_blocks, block_size, H, D)
    v_cache = torch.randn(num_blocks, block_size, H, D)
    block_tables = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    seq_lens = torch.tensor([25, 30], dtype=torch.int32)

    out = paged_attention_kernel(q, k_cache, v_cache, block_tables, seq_lens, block_size=block_size)
    ref = paged_attention_pytorch(q, k_cache, v_cache, block_tables, seq_lens, block_size=block_size)

    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
    assert (out - ref).abs().max().item() < 1e-5


def test_gla_scan_parity():
    """Verify GLA recurrent scan matches pure PyTorch reference."""
    from neuralops.ops.triton.gla import gla_scan, gla_scan_pytorch

    B, H, L, D = 2, 4, 16, 32
    q = torch.randn(B, H, L, D)
    k = torch.randn(B, H, L, D)
    v = torch.randn(B, H, L, D)
    lam = torch.sigmoid(torch.randn(B, H, L))

    out = gla_scan(q, k, v, lam)
    ref = gla_scan_pytorch(q, k, v, lam)

    assert not torch.isnan(out).any()
    assert (out - ref).abs().max().item() < 1e-5


def test_gdn_scan_parity():
    """Verify Gated DeltaNet scan matches pure PyTorch reference."""
    from neuralops.ops.triton.gdn import gdn_scan_pytorch, gdn_scan_triton

    B, H, L, D = 2, 4, 16, 32
    q = torch.randn(B, H, L, D)
    k = torch.randn(B, H, L, D)
    v = torch.randn(B, H, L, D)
    beta = torch.sigmoid(torch.randn(B, H, L))
    alpha = torch.sigmoid(torch.randn(B, H, L))

    out, _ = gdn_scan_triton(q, k, v, beta, alpha)
    ref, _ = gdn_scan_pytorch(q, k, v, beta, alpha)

    assert not torch.isnan(out).any()
    assert (out - ref).abs().max().item() < 1e-5


def test_fused_linear_cross_entropy_parity():
    """Verify FusedLinearCrossEntropy matches un-fused linear + cross_entropy."""
    from neuralops.ops.triton.fused_linear_cross_entropy import (
        fused_linear_cross_entropy,
    )

    h = torch.randn(4, 32, requires_grad=True)
    w = torch.randn(50, 32, requires_grad=True)
    b = torch.randn(50, requires_grad=True)
    targets = torch.randint(0, 50, (4,))

    loss = fused_linear_cross_entropy(h, w, targets, bias=b)
    ref_loss = F.cross_entropy(F.linear(h, w, b), targets)

    assert not torch.isnan(loss).any()
    assert (loss - ref_loss).abs().item() < 1e-5

    loss.backward()
    assert h.grad is not None and not torch.isnan(h.grad).any()
    assert w.grad is not None and not torch.isnan(w.grad).any()


def test_fused_linear_kl_divergence_parity():
    """Verify FusedLinearKLDivergence matches un-fused linear + KL divergence."""
    from neuralops.ops.triton.kldiv import (
        fused_kd_loss_pytorch,
        fused_linear_kl_divergence,
    )

    s_h = torch.randn(4, 32, requires_grad=True)
    s_w = torch.randn(50, 32, requires_grad=True)
    s_b = torch.randn(50, requires_grad=True)
    t_h = torch.randn(4, 32)
    t_w = torch.randn(50, 32)
    t_b = torch.randn(50)

    loss = fused_linear_kl_divergence(
        s_h,
        s_w,
        t_h,
        t_w,
        student_bias=s_b,
        teacher_bias=t_b,
        temperature=2.0,
    )
    s_logits = F.linear(s_h, s_w, s_b)
    t_logits = F.linear(t_h, t_w, t_b)
    ref_loss = fused_kd_loss_pytorch(s_logits, t_logits, temperature=2.0)

    assert not torch.isnan(loss).any()
    assert (loss - ref_loss).abs().item() < 1e-5

    loss.backward()
    assert s_h.grad is not None and not torch.isnan(s_h.grad).any()


def test_standalone_fused_kronecker_matmul():
    """Verify Kronecker GEMM matches exact mathematical Kronecker factorization."""
    from neuralops.ops.triton.standalone.fused_kronecker_matmul import (
        triton_fused_kronecker_matmul,
        triton_fused_kronecker_matmul_pytorch_ref,
    )

    M, N1, N2 = 4, 4, 4
    M1, M2 = 4, 4
    x = torch.randn(M, N1 * N2)
    A = torch.randn(M1, N1)
    B = torch.randn(M2, N2)

    out = triton_fused_kronecker_matmul(x, A, B)
    ref = triton_fused_kronecker_matmul_pytorch_ref(x, A, B)

    # True Kronecker product
    K_full = torch.kron(A, B)
    math_ref = torch.matmul(x, K_full.t())

    assert not torch.isnan(out).any()
    assert (out - ref).abs().max().item() < 1e-5
    assert (out - math_ref).abs().max().item() < 1e-5


def test_standalone_act_scale_quant_gemm():
    """Verify fused activation-scale GEMM matches reference."""
    from neuralops.ops.triton.standalone import (
        triton_fused_act_scale_quant_gemm,
        triton_fused_act_scale_quant_gemm_pytorch_ref,
    )

    x = torch.randn(4, 32)
    scale = torch.rand(32) + 0.1
    w = torch.randn(64, 32)
    bias = torch.randn(64)

    out = triton_fused_act_scale_quant_gemm(x, scale, w, bias)
    ref = triton_fused_act_scale_quant_gemm_pytorch_ref(x, scale, w, bias)

    assert not torch.isnan(out).any()
    assert (out - ref).abs().max().item() < 1e-5


def test_standalone_hadamard_w4a16_gemm():
    """Verify fused Hadamard W4A16 GEMM matches reference."""
    from neuralops.ops.triton.fused_norm_linear import (
        triton_fused_hadamard_w4a16_gemm,
        triton_fused_hadamard_w4a16_gemm_pytorch_ref,
    )

    M, K, N = 4, 32, 16
    group_size = 16
    x = torch.randn(M, K)
    w_packed = torch.randint(0, 255, (N, K // 2), dtype=torch.uint8)
    scales = torch.rand(N, K // group_size) + 0.1
    zeros = torch.zeros(N, K // group_size)

    out = triton_fused_hadamard_w4a16_gemm(x, w_packed, scales, zeros, group_size=group_size)
    ref = triton_fused_hadamard_w4a16_gemm_pytorch_ref(x, w_packed, scales, zeros, group_size=group_size)

    assert not torch.isnan(out).any()
    assert (out - ref).abs().max().item() < 1e-5


def test_standalone_lut_codebook_dequant_gemm():
    """Verify in-register codebook LUT GEMM matches reference."""
    from neuralops.ops.triton.fused_norm_linear import (
        triton_lut_codebook_dequant_gemm,
        triton_lut_codebook_dequant_gemm_pytorch_ref,
    )

    M, K, N = 4, 32, 16
    group_size = 16
    x = torch.randn(M, K)
    codes = torch.randint(0, 16, (N, K // 2), dtype=torch.uint8)
    codebook = torch.randn(K // group_size, 16)

    out = triton_lut_codebook_dequant_gemm(x, codes, codebook, group_size=group_size)
    ref = triton_lut_codebook_dequant_gemm_pytorch_ref(x, codes, codebook, group_size=group_size)

    assert not torch.isnan(out).any()
    assert (out - ref).abs().max().item() < 1e-5


def test_standalone_latent_kv_cache_attention():
    """Verify Latent-Space FlashAttention matches reference."""
    from neuralops.ops.triton.standalone import (
        triton_fused_latent_kv_cache_attention,
        triton_fused_latent_kv_cache_attention_pytorch_ref,
    )

    B, H, N, D_latent, D_out = 2, 4, 16, 32, 64
    q = torch.randn(B, H, N, D_out)
    latent_kv = torch.randn(B, N, D_latent)
    W_k = torch.randn(D_out, D_latent)
    W_v = torch.randn(D_latent, D_out)

    out = triton_fused_latent_kv_cache_attention(q, latent_kv, W_k, W_v)
    ref = triton_fused_latent_kv_cache_attention_pytorch_ref(q, latent_kv, W_k, W_v)

    assert not torch.isnan(out).any()
    assert (out - ref).abs().max().item() < 1e-5


def test_extended_kernels_sampling():
    """Sample and verify extended Triton kernels."""
    from neuralops.ops.triton.cosformer_triton_kernel import (
        triton_cosformer_linear_attention,
        triton_cosformer_linear_attention_pytorch_ref,
    )
    from neuralops.ops.triton.cur_linear import (
        triton_cur_linear,
        triton_cur_linear_pytorch_ref,
    )
    from neuralops.ops.triton.differential_attention import (
        triton_differential_attention,
        triton_differential_attention_pytorch_ref,
    )
    from neuralops.ops.triton.dynamic_sparse_attention import (
        triton_dynamic_sparse_attention,
        triton_dynamic_sparse_attention_pytorch_ref,
    )
    from neuralops.ops.triton.fms_scan import (
        fms_scan_pytorch,
        fms_scan_triton,
    )
    from neuralops.ops.triton.hgrn2_scan import (
        hgrn2_scan_pytorch,
        hgrn2_scan_triton,
    )
    from neuralops.ops.triton.hydra_attention import (
        triton_hydra_attention,
        triton_hydra_attention_pytorch_ref,
    )
    from neuralops.ops.triton.low_rank import fused_lowrank_gemm
    from neuralops.ops.triton.retnet_retention import (
        triton_retnet_retention,
        triton_retnet_retention_pytorch_ref,
    )
    from neuralops.ops.triton.selective_scan import (
        selective_scan_fn,
        selective_scan_ref,
    )
    from neuralops.ops.triton.sliding_window import (
        sliding_window_attention,
        sliding_window_pytorch,
    )

    # cur_linear
    x = torch.randn(2, 16, 64)
    R = torch.randn(16, 64)
    U = torch.randn(16, 16)
    C = torch.randn(32, 16)
    assert not torch.isnan(triton_cur_linear(x, R, U, C)).any()
    assert (triton_cur_linear(x, R, U, C) - triton_cur_linear_pytorch_ref(x, R, U, C)).abs().max().item() < 1e-5

    # differential_attention
    q1, q2 = torch.randn(2, 4, 16, 32), torch.randn(2, 4, 16, 32)
    k1, k2 = torch.randn(2, 4, 16, 32), torch.randn(2, 4, 16, 32)
    v = torch.randn(2, 4, 16, 32)
    da_out = triton_differential_attention(q1, q2, k1, k2, v)
    da_ref = triton_differential_attention_pytorch_ref(q1, q2, k1, k2, v)
    assert not torch.isnan(da_out).any()
    assert (da_out - da_ref).abs().max().item() < 1e-5

    # cosformer
    q = torch.randn(2, 4, 16, 32)
    k = torch.randn(2, 4, 16, 32)
    cos_out = triton_cosformer_linear_attention(q, k, v)
    cos_ref = triton_cosformer_linear_attention_pytorch_ref(q, k, v)
    assert not torch.isnan(cos_out).any()
    assert (cos_out - cos_ref).abs().max().item() < 1e-5

    # hydra
    hydra_out = triton_hydra_attention(q, k, v)
    hydra_ref = triton_hydra_attention_pytorch_ref(q, k, v)
    assert not torch.isnan(hydra_out).any()
    assert (hydra_out - hydra_ref).abs().max().item() < 1e-5

    # sliding_window
    sw_out = sliding_window_attention(q, k, v, window_size=8)
    sw_ref = sliding_window_pytorch(q, k, v, window_size=8)
    assert not torch.isnan(sw_out).any()
    assert (sw_out - sw_ref).abs().max().item() < 1e-5

    # selective_scan
    B_s, D_s, L_s, N_s = 2, 8, 16, 4
    u_ss = torch.randn(B_s, D_s, L_s)
    delta = torch.randn(B_s, D_s, L_s).sigmoid()
    A_ss = -torch.rand(D_s, N_s)
    B_mat = torch.randn(B_s, N_s, L_s)
    C_mat = torch.randn(B_s, N_s, L_s)
    ss_out = selective_scan_fn(u_ss, delta, A_ss, B_mat, C_mat)
    ss_ref = selective_scan_ref(u_ss, delta, A_ss, B_mat, C_mat)
    assert not torch.isnan(ss_out).any()
    assert (ss_out - ss_ref).abs().max().item() < 1e-5

    # retnet_retention
    gammas = torch.rand(4)
    ret_out = triton_retnet_retention(q, k, v, gammas)
    ret_ref = triton_retnet_retention_pytorch_ref(q, k, v, gammas)
    assert not torch.isnan(ret_out).any()
    assert (ret_out - ret_ref).abs().max().item() < 1e-5

    # fms_scan
    q_fms = torch.randn(2, 4, 16, 32)
    k_fms = torch.randn(2, 4, 16, 32)
    v_fms = torch.randn(2, 4, 16, 32)
    decay_fms = torch.sigmoid(torch.randn(4))
    fms_out = fms_scan_triton(q_fms, k_fms, v_fms, decay_fms)
    fms_ref = fms_scan_pytorch(q_fms, k_fms, v_fms, decay_fms)
    assert not torch.isnan(fms_out).any()
    assert (fms_out - fms_ref).abs().max().item() < 1e-5

    # hgrn2_scan
    x_h = torch.randn(2, 16, 32)
    g_h = torch.sigmoid(torch.randn(2, 16, 32))
    assert not torch.isnan(hgrn2_scan_triton(x_h, g_h)).any()
    assert (hgrn2_scan_triton(x_h, g_h) - hgrn2_scan_pytorch(x_h, g_h)).abs().max().item() < 1e-5

    # low_rank
    x_lr = torch.randn(2, 32)
    u_lr = torch.randn(32, 16)
    v_lr = torch.randn(16, 32)
    lr_out = fused_lowrank_gemm(x_lr, u_lr, v_lr)
    assert not torch.isnan(lr_out).any()
    assert (lr_out - (x_lr @ u_lr @ v_lr)).abs().max().item() < 1e-5


def test_autotuner_and_sram_budgets():
    """Verify SRAM budget estimation, safe autotuning filtering, and constants."""
    assert BLOCK_SIZE == 64
    assert MAX_TRITON_SEQ_LEN == 16384
    assert 64 in SUPPORTED_HEAD_DIMS

    # Estimate SRAM bytes
    sram_fp16 = estimate_kernel_sram_bytes(block_m=64, block_n=64, head_dim=64, dtype_bytes=2, num_stages=2)
    assert sram_fp16 > 0
    assert isinstance(sram_fp16, int)

    # Safe autotune configs for CPU or default profile
    safe_cfgs = get_safe_autotune_configs(head_dim=64, dtype=torch.float16)
    assert len(safe_cfgs) > 0
    for cfg in safe_cfgs:
        assert "BLOCK_M" in cfg
        assert "BLOCK_N" in cfg
        assert "num_warps" in cfg


def test_governance_no_deprecated_amp():
    """Governance test: verify zero imports of deprecated torch.cuda.amp across triton ops (Rule 2.1)."""
    triton_dir = Path(__file__).resolve().parents[2] / "neuralops" / "ops" / "triton"
    py_files = sorted(triton_dir.glob("**/*.py"))
    assert len(py_files) >= 50, f"Expected >= 50 python files in ops/triton, found {len(py_files)}"

    violations = []
    for f in py_files:
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if "torch.cuda.amp" in alias.name:
                        violations.append((str(f), alias.name))
            elif isinstance(node, ast.ImportFrom):
                if node.module and "torch.cuda.amp" in node.module:
                    violations.append((str(f), node.module))

    assert not violations, f"Forbidden torch.cuda.amp imports found: {violations}"
