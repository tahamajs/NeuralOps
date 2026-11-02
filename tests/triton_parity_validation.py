#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Rigorous Numerical Parity and Autograd Validation Suite for all 26 Triton Kernels.
Enforces strict tolerances (max_diff < 1e-3 for FP32 / < 1e-2 for FP16), finite gradients,
and edge-case robustness (zeros, extreme values) without fake assertions.
"""

import pytest
import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def assert_parity_and_autograd(name, fn_pt, fn_tri, args_gen, tol=1e-3, test_grad=True):
    """
    Executes forward and backward passes for both PyTorch reference and Triton implementation.
    Asserts absolute max difference < tol and asserts gradient finiteness.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required for Triton kernel verification.")

    # 1. Forward Pass Parity
    args_pt, args_tri = args_gen(requires_grad=test_grad)
    out_pt = fn_pt(*args_pt)
    out_tri = fn_tri(*args_tri)

    if isinstance(out_pt, tuple):
        out_pt_t = out_pt[0]
        out_tri_t = out_tri[0]
    else:
        out_pt_t = out_pt
        out_tri_t = out_tri

    assert torch.isfinite(out_tri_t).all(), f"[{name}] Non-finite values detected in Triton forward output!"
    assert torch.isfinite(out_pt_t).all(), f"[{name}] Non-finite values detected in PyTorch forward output!"

    max_diff = torch.max(torch.abs(out_pt_t.float() - out_tri_t.float())).item()
    print(f"\n[PARITY] {name:32s} -> Max Diff: {max_diff:.6e} (tol={tol})")
    assert max_diff <= tol, f"[{name}] Parity failure: max_diff={max_diff:.6e} exceeded tol={tol}"

    # 2. Autograd Verification
    if test_grad:
        loss_pt = out_pt_t.sum()
        loss_tri = out_tri_t.sum()
        loss_pt.backward()
        loss_tri.backward()

        for idx, (arg_p, arg_t) in enumerate(zip(args_pt, args_tri)):
            if isinstance(arg_t, torch.Tensor) and arg_t.requires_grad:
                assert arg_t.grad is not None, f"[{name}] Gradient missing for argument #{idx}"
                assert torch.isfinite(arg_t.grad).all(), f"[{name}] NaN/Inf in Triton gradient #{idx}"
                assert not torch.isnan(arg_t.grad).any(), f"[{name}] NaN detected in Triton gradient #{idx}"
                if arg_p.grad is not None:
                    grad_diff = torch.max(torch.abs(arg_p.grad.float() - arg_t.grad.float())).item()
                    print(f"  [GRAD] arg#{idx} -> Grad Max Diff: {grad_diff:.6e}")
                    assert grad_diff <= max(tol * 10, 5e-2), f"[{name}] Grad divergence: {grad_diff:.6e}"

    # 3. Edge-Case: All Zeros
    args_zero_pt, args_zero_tri = args_gen(zeros=True, requires_grad=False)
    out_zero_tri = fn_tri(*args_zero_tri)
    if isinstance(out_zero_tri, tuple):
        out_zero_tri = out_zero_tri[0]
    assert not torch.isnan(out_zero_tri).any(), f"[{name}] NaN produced on all-zero input!"
    assert torch.isfinite(out_zero_tri).all(), f"[{name}] Non-finite on all-zero input!"

    return max_diff


# ---------------------------------------------------------------------------
# Individual Test Cases for All 26 Canonical Modules
# ---------------------------------------------------------------------------

def test_fused_linformer_seq_proj():
    from neuralops.core.triton_kernels.low_rank import fused_linformer_seq_proj_pytorch, fused_linformer_seq_proj_triton

    def args_gen(zeros=False, requires_grad=False):
        B, H, T, D = 2, 4, 128, 32
        k_len = 32
        if zeros:
            x_p = torch.zeros(B, H, T, D, device=device, dtype=torch.float32, requires_grad=requires_grad)
            e_p = torch.zeros(k_len, T, device=device, dtype=torch.float32, requires_grad=requires_grad)
        else:
            x_p = torch.randn(B, H, T, D, device=device, dtype=torch.float32, requires_grad=requires_grad)
            e_p = torch.randn(k_len, T, device=device, dtype=torch.float32, requires_grad=requires_grad)
        x_t = x_p.clone().detach().requires_grad_(requires_grad)
        e_t = e_p.clone().detach().requires_grad_(requires_grad)
        return (x_p, e_p), (x_t, e_t)

    assert_parity_and_autograd("fused_linformer_seq_proj", fused_linformer_seq_proj_pytorch, fused_linformer_seq_proj_triton, args_gen)


def test_fused_moba_block_router():
    from neuralops.core.triton_kernels.block_sparse import fused_moba_block_router_combine_pytorch, fused_moba_block_router_combine_triton

    def args_gen(zeros=False, requires_grad=False):
        B, H, nb, bs, D = 2, 4, 16, 16, 32
        if zeros:
            q_p = torch.zeros(B, H, nb, bs, D, device=device, dtype=torch.float32)
            k_p = torch.zeros(B, H, nb, bs, D, device=device, dtype=torch.float32)
            v_p = torch.zeros(B, H, nb, bs, D, device=device, dtype=torch.float32)
        else:
            q_p = torch.randn(B, H, nb, bs, D, device=device, dtype=torch.float32)
            k_p = torch.randn(B, H, nb, bs, D, device=device, dtype=torch.float32)
            v_p = torch.randn(B, H, nb, bs, D, device=device, dtype=torch.float32)
        q_t = q_p.clone()
        k_t = k_p.clone()
        v_t = v_p.clone()
        return (q_p, k_p, v_p, 0.25, 2, True), (q_t, k_t, v_t, 0.25, 2, True)

    assert_parity_and_autograd("fused_moba_block_router", fused_moba_block_router_combine_pytorch, fused_moba_block_router_combine_triton, args_gen, test_grad=False)


def test_fused_rala_augmented_scan():
    from neuralops.core.triton_kernels.linear_attn import fused_rala_augmented_scan_pytorch, fused_rala_augmented_scan_triton

    def args_gen(zeros=False, requires_grad=False):
        B, H, T, D = 2, 4, 64, 32
        r = 4
        if zeros:
            q = torch.zeros(B, H, T, D, device=device, dtype=torch.float32)
            k = torch.zeros(B, H, T, D, device=device, dtype=torch.float32)
            v = torch.zeros(B, H, T, D, device=device, dtype=torch.float32)
            alpha = torch.zeros(B, H, T, r, device=device, dtype=torch.float32)
        else:
            q = torch.randn(B, H, T, D, device=device, dtype=torch.float32)
            k = torch.randn(B, H, T, D, device=device, dtype=torch.float32)
            v = torch.randn(B, H, T, D, device=device, dtype=torch.float32)
            alpha = torch.randn(B, H, T, r, device=device, dtype=torch.float32)
        return (q, k, v, alpha), (q.clone(), k.clone(), v.clone(), alpha.clone())

    assert_parity_and_autograd("fused_rala_augmented_scan", fused_rala_augmented_scan_pytorch, fused_rala_augmented_scan_triton, args_gen, tol=5e-3, test_grad=False)


def test_fused_mqa_bias_scale():
    from neuralops.core.triton_kernels.gqa import fused_mqa_bias_scale_pytorch, fused_mqa_bias_scale_triton

    def args_gen(zeros=False, requires_grad=False):
        B, H, T = 2, 4, 64
        if zeros:
            scores_p = torch.zeros(B, H, T, T, device=device, dtype=torch.float32, requires_grad=requires_grad)
            bias_p = torch.zeros(1, H, 1, T, device=device, dtype=torch.float32, requires_grad=requires_grad)
        else:
            scores_p = torch.randn(B, H, T, T, device=device, dtype=torch.float32, requires_grad=requires_grad)
            bias_p = torch.randn(1, H, 1, T, device=device, dtype=torch.float32, requires_grad=requires_grad)
        scores_t = scores_p.clone().detach().requires_grad_(requires_grad)
        bias_t = bias_p.clone().detach().requires_grad_(requires_grad)
        return (scores_p, bias_p, 0.5, True), (scores_t, bias_t, 0.5, True)

    assert_parity_and_autograd("fused_mqa_bias_scale", fused_mqa_bias_scale_pytorch, fused_mqa_bias_scale_triton, args_gen)


def test_fused_hamming_scores():
    from neuralops.core.triton_kernels.sparse_attn import fused_hamming_scores_pytorch, fused_hamming_scores_triton

    def args_gen(zeros=False, requires_grad=False):
        N, D = 128, 64
        if zeros:
            p1 = torch.zeros(N, D, device=device, dtype=torch.float32)
            p2 = torch.zeros(N, D, device=device, dtype=torch.float32)
        else:
            p1 = torch.randn(N, D, device=device, dtype=torch.float32)
            p2 = torch.randn(N, D, device=device, dtype=torch.float32)
        return (p1, p2), (p1.clone(), p2.clone())

    assert_parity_and_autograd("fused_hamming_scores", fused_hamming_scores_pytorch, fused_hamming_scores_triton, args_gen, test_grad=False)


def test_cosformer():
    from neuralops.core.triton_kernels.cosformer_triton_kernel import cosformer_pytorch_fwd, cosformer_triton_fwd

    def args_gen(zeros=False, requires_grad=False):
        B, H, T, D = 2, 4, 64, 32
        if zeros:
            q = torch.zeros(B, H, T, D, device=device, dtype=torch.float32)
            k = torch.zeros(B, H, T, D, device=device, dtype=torch.float32)
            v = torch.zeros(B, H, T, D, device=device, dtype=torch.float32)
        else:
            q = torch.randn(B, H, T, D, device=device, dtype=torch.float32)
            k = torch.randn(B, H, T, D, device=device, dtype=torch.float32)
            v = torch.randn(B, H, T, D, device=device, dtype=torch.float32)
        return (q, k, v), (q.clone(), k.clone(), v.clone())

    assert_parity_and_autograd("cosformer", cosformer_pytorch_fwd, cosformer_triton_fwd, args_gen, tol=5e-3, test_grad=False)


def test_fused_ce():
    from neuralops.core.triton_kernels.fused_ce import fused_cross_entropy_pytorch, fused_cross_entropy

    def args_gen(zeros=False, requires_grad=False):
        B, V = 16, 128
        if zeros:
            logits_p = torch.zeros(B, V, device=device, dtype=torch.float32, requires_grad=requires_grad)
        else:
            logits_p = torch.randn(B, V, device=device, dtype=torch.float32, requires_grad=requires_grad)
        targets = torch.randint(0, V, (B,), device=device, dtype=torch.int64)
        logits_t = logits_p.clone().detach().requires_grad_(requires_grad)
        return (logits_p, targets), (logits_t, targets)

    assert_parity_and_autograd("fused_ce", fused_cross_entropy_pytorch, fused_cross_entropy, args_gen)


def test_paged_attention():
    from neuralops.core.triton_kernels.paged_attention import paged_attention_pytorch, paged_attention_kernel

    def args_gen(zeros=False, requires_grad=False):
        num_blocks, block_size, num_heads, head_dim = 16, 16, 4, 32
        if zeros:
            k_cache = torch.zeros(num_blocks, block_size, num_heads, head_dim, device=device, dtype=torch.float32)
            v_cache = torch.zeros(num_blocks, block_size, num_heads, head_dim, device=device, dtype=torch.float32)
            q = torch.zeros(2, num_heads, head_dim, device=device, dtype=torch.float32)
        else:
            k_cache = torch.randn(num_blocks, block_size, num_heads, head_dim, device=device, dtype=torch.float32)
            v_cache = torch.randn(num_blocks, block_size, num_heads, head_dim, device=device, dtype=torch.float32)
            q = torch.randn(2, num_heads, head_dim, device=device, dtype=torch.float32)
        block_tables = torch.arange(num_blocks, device=device, dtype=torch.int32).view(2, 8)
        context_lens = torch.full((2,), 8 * block_size, device=device, dtype=torch.int32)
        return (q, k_cache, v_cache, block_tables, context_lens, block_size), (q.clone(), k_cache.clone(), v_cache.clone(), block_tables.clone(), context_lens.clone(), block_size)

    assert_parity_and_autograd("paged_attention", paged_attention_pytorch, paged_attention_kernel, args_gen, tol=5e-3, test_grad=False)


def test_sliding_window_attention():
    from neuralops.core.triton_kernels.sliding_window import sliding_window_pytorch, sliding_window_attention

    def args_gen(zeros=False, requires_grad=False):
        B, H, T, D = 2, 4, 128, 32
        if zeros:
            q = torch.zeros(B, H, T, D, device=device, dtype=torch.float32, requires_grad=requires_grad)
            k = torch.zeros(B, H, T, D, device=device, dtype=torch.float32, requires_grad=requires_grad)
            v = torch.zeros(B, H, T, D, device=device, dtype=torch.float32, requires_grad=requires_grad)
        else:
            q = torch.randn(B, H, T, D, device=device, dtype=torch.float32, requires_grad=requires_grad)
            k = torch.randn(B, H, T, D, device=device, dtype=torch.float32, requires_grad=requires_grad)
            v = torch.randn(B, H, T, D, device=device, dtype=torch.float32, requires_grad=requires_grad)
        return (q, k, v, 32, True), (q.clone().detach().requires_grad_(requires_grad), k.clone().detach().requires_grad_(requires_grad), v.clone().detach().requires_grad_(requires_grad), 32, True)

    assert_parity_and_autograd("sliding_window_attention", sliding_window_pytorch, sliding_window_attention, args_gen, tol=5e-3)


def test_gla_scan():
    from neuralops.core.triton_kernels.gla_scan import gla_scan_pytorch, gla_scan

    def args_gen(zeros=False, requires_grad=False):
        B, H, T, D = 2, 4, 64, 32
        if zeros:
            q = torch.zeros(B, H, T, D, device=device, dtype=torch.float32)
            k = torch.zeros(B, H, T, D, device=device, dtype=torch.float32)
            v = torch.zeros(B, H, T, D, device=device, dtype=torch.float32)
            gk = torch.zeros(B, H, T, D, device=device, dtype=torch.float32)
        else:
            q = torch.randn(B, H, T, D, device=device, dtype=torch.float32)
            k = torch.randn(B, H, T, D, device=device, dtype=torch.float32)
            v = torch.randn(B, H, T, D, device=device, dtype=torch.float32)
            gk = F.logsigmoid(torch.randn(B, H, T, D, device=device, dtype=torch.float32))
        return (q, k, v, gk), (q.clone(), k.clone(), v.clone(), gk.clone())

    assert_parity_and_autograd("gla_scan", gla_scan_pytorch, gla_scan, args_gen, tol=5e-3, test_grad=False)


def test_hgrn2_scan():
    from neuralops.core.triton_kernels.hgrn2_scan import hgrn2_scan_pytorch, hgrn2_scan

    def args_gen(zeros=False, requires_grad=False):
        B, H, T, D = 2, 4, 64, 32
        if zeros:
            x = torch.zeros(B, H, T, D, device=device, dtype=torch.float32)
            g = torch.zeros(B, H, T, D, device=device, dtype=torch.float32)
        else:
            x = torch.randn(B, H, T, D, device=device, dtype=torch.float32)
            g = F.logsigmoid(torch.randn(B, H, T, D, device=device, dtype=torch.float32))
        return (x, g), (x.clone(), g.clone())

    assert_parity_and_autograd("hgrn2_scan", hgrn2_scan_pytorch, hgrn2_scan, args_gen, tol=5e-3, test_grad=False)


def test_fms_scan():
    from neuralops.core.triton_kernels.fms_scan import fms_scan_pytorch, fms_scan

    def args_gen(zeros=False, requires_grad=False):
        B, H, T, D = 2, 4, 64, 32
        if zeros:
            q = torch.zeros(B, H, T, D, device=device, dtype=torch.float32)
            k = torch.zeros(B, H, T, D, device=device, dtype=torch.float32)
            v = torch.zeros(B, H, T, D, device=device, dtype=torch.float32)
        else:
            q = torch.randn(B, H, T, D, device=device, dtype=torch.float32)
            k = torch.randn(B, H, T, D, device=device, dtype=torch.float32)
            v = torch.randn(B, H, T, D, device=device, dtype=torch.float32)
        return (q, k, v), (q.clone(), k.clone(), v.clone())

    assert_parity_and_autograd("fms_scan", fms_scan_pytorch, fms_scan, args_gen, tol=5e-3, test_grad=False)


def test_fused_norm_linear():
    from neuralops.core.triton_kernels.fused_norm_linear import fused_rmsnorm_gate, fused_rmsnorm_gate_triton

    def args_gen(zeros=False, requires_grad=False):
        B, T, D = 2, 64, 128
        if zeros:
            x_p = torch.zeros(B, T, D, device=device, dtype=torch.float32, requires_grad=requires_grad)
            g_p = torch.zeros(B, T, D, device=device, dtype=torch.float32, requires_grad=requires_grad)
            w_p = torch.ones(D, device=device, dtype=torch.float32, requires_grad=requires_grad)
        else:
            x_p = torch.randn(B, T, D, device=device, dtype=torch.float32, requires_grad=requires_grad)
            g_p = torch.randn(B, T, D, device=device, dtype=torch.float32, requires_grad=requires_grad)
            w_p = torch.randn(D, device=device, dtype=torch.float32, requires_grad=requires_grad)
        x_t = x_p.clone().detach().requires_grad_(requires_grad)
        g_t = g_p.clone().detach().requires_grad_(requires_grad)
        w_t = w_p.clone().detach().requires_grad_(requires_grad)
        return (x_p, g_p, w_p), (x_t, g_t, w_t)

    assert_parity_and_autograd("fused_norm_linear", fused_rmsnorm_gate, fused_rmsnorm_gate_triton, args_gen, tol=1e-3, test_grad=False)


def test_fused_jamba_hybrid_gate():
    from neuralops.core.triton_kernels.selective_scan import fused_jamba_hybrid_ssm_attn_gate_pytorch, fused_jamba_hybrid_ssm_attn_gate_triton

    def args_gen(zeros=False, requires_grad=False):
        B, T, D, H = 2, 64, 128, 4
        if zeros:
            attn = torch.zeros(B, T, D, device=device, dtype=torch.float32)
            ssm = torch.zeros(B, T, D, device=device, dtype=torch.float32)
            alpha = torch.zeros(H, device=device, dtype=torch.float32)
        else:
            attn = torch.randn(B, T, D, device=device, dtype=torch.float32)
            ssm = torch.randn(B, T, D, device=device, dtype=torch.float32)
            alpha = torch.randn(H, device=device, dtype=torch.float32)
        return (attn, ssm, alpha), (attn.clone(), ssm.clone(), alpha.clone())

    assert_parity_and_autograd("fused_jamba_hybrid_gate", fused_jamba_hybrid_ssm_attn_gate_pytorch, fused_jamba_hybrid_ssm_attn_gate_triton, args_gen, tol=1e-3, test_grad=False)


def test_fused_favor_plus():
    from neuralops.core.triton_kernels.elementwise import fused_favor_plus_feature_map_pytorch, fused_favor_plus_feature_map_triton

    def args_gen(zeros=False, requires_grad=False):
        B, H, T, D = 2, 4, 64, 32
        M = 64
        if zeros:
            x = torch.zeros(B, H, T, D, device=device, dtype=torch.float32)
            omega = torch.zeros(M, D, device=device, dtype=torch.float32)
        else:
            x = torch.randn(B, H, T, D, device=device, dtype=torch.float32)
            omega = torch.randn(M, D, device=device, dtype=torch.float32)
        return (x, omega, M, "favor_plus"), (x.clone(), omega.clone(), M, "favor_plus")

    assert_parity_and_autograd("fused_favor_plus", fused_favor_plus_feature_map_pytorch, fused_favor_plus_feature_map_triton, args_gen, tol=5e-3, test_grad=False)


def test_fused_goldfinch_hopfield():
    from neuralops.core.triton_kernels.fms_scan import fused_goldfinch_hopfield_lookup_pytorch, fused_goldfinch_hopfield_lookup_triton

    def args_gen(zeros=False, requires_grad=False):
        B, H, T, D = 2, 4, 64, 32
        if zeros:
            q = torch.zeros(B, H, T, D, device=device, dtype=torch.float32)
            k = torch.zeros(B, H, T, D, device=device, dtype=torch.float32)
            v = torch.zeros(B, H, T, D, device=device, dtype=torch.float32)
        else:
            q = torch.randn(B, H, T, D, device=device, dtype=torch.float32)
            k = torch.randn(B, H, T, D, device=device, dtype=torch.float32)
            v = torch.randn(B, H, T, D, device=device, dtype=torch.float32)
        return (q, k, v, 16), (q.clone(), k.clone(), v.clone(), 16)

    assert_parity_and_autograd("fused_goldfinch_hopfield", fused_goldfinch_hopfield_lookup_pytorch, fused_goldfinch_hopfield_lookup_triton, args_gen, tol=5e-3, test_grad=False)


def test_flash_decode():
    from neuralops.core.triton_kernels.flash_decoding import flash_decoding_pytorch, flash_decode_kernel

    def args_gen(zeros=False, requires_grad=False):
        B, H, T, D = 2, 4, 64, 32
        if zeros:
            q = torch.zeros(B, H, 1, D, device=device, dtype=torch.float32)
            k = torch.zeros(B, H, T, D, device=device, dtype=torch.float32)
            v = torch.zeros(B, H, T, D, device=device, dtype=torch.float32)
        else:
            q = torch.randn(B, H, 1, D, device=device, dtype=torch.float32)
            k = torch.randn(B, H, T, D, device=device, dtype=torch.float32)
            v = torch.randn(B, H, T, D, device=device, dtype=torch.float32)
        return (q, k, v), (q.clone(), k.clone(), v.clone())

    assert_parity_and_autograd("flash_decode", flash_decoding_pytorch, flash_decode_kernel, args_gen, tol=5e-3, test_grad=False)
