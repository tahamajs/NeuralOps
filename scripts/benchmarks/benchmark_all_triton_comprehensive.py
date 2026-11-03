#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Hardware Benchmark for All Available Triton Modules in NeuralOps.
Measures Triton latency, PyTorch baseline latency, Speedup, Max Diff, and Status.
"""

import os
import sys
import time
import math
import csv
import json
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

def time_fn(fn, *args, warmups=10, iters=30, **kwargs):
    if device == "cuda":
        for _ in range(warmups):
            fn(*args, **kwargs)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            fn(*args, **kwargs)
        torch.cuda.synchronize()
        end = time.perf_counter()
    else:
        for _ in range(warmups):
            fn(*args, **kwargs)
        start = time.perf_counter()
        for _ in range(iters):
            fn(*args, **kwargs)
        end = time.perf_counter()
    return (end - start) / iters * 1000.0

def run_benchmarks():
    print(f"Benchmarking All Triton Modules on device: {device}, dtype: {dtype}")
    results = []

    def record(name, cat, t_tri, t_pt, diff, speedup=None):
        if speedup is None:
            speedup = t_pt / max(t_tri, 1e-6)
        status = "PASS" if speedup >= 1.0 and diff < 2e-2 else ("SLOW" if diff < 2e-2 else "FAIL")
        results.append({
            "module": name, "category": cat, "triton_ms": t_tri,
            "pytorch_ms": t_pt, "speedup": speedup, "max_diff": diff, "status": status
        })
        print(f"[{status:<4}] {name:<32} | Tri: {t_tri:7.3f}ms | PT: {t_pt:7.3f}ms | Speedup: {speedup:6.2f}x | Diff: {diff:.2e}")

    # 1. Softplus
    try:
        from neuralops.ops.triton.softplus import softplus_fwd, softplus_pytorch
        x = torch.randn(1024, 2048, device=device, dtype=dtype)
        t_tri = time_fn(softplus_fwd, x)
        t_pt = time_fn(softplus_pytorch, x)
        diff = (softplus_fwd(x) - softplus_pytorch(x)).abs().max().item()
        record("softplus", "elementwise", t_tri, t_pt, diff)
    except Exception as e:
        print(f"softplus error: {e}")

    # 2. Fused Cross Entropy
    try:
        from neuralops.ops.triton.fused_ce import fused_cross_entropy, fused_cross_entropy_pytorch
        logits = torch.randn(256, 8192, device=device, dtype=dtype)
        targets = torch.randint(0, 8192, (256,), device=device, dtype=torch.long)
        t_tri = time_fn(fused_cross_entropy, logits, targets)
        t_pt = time_fn(fused_cross_entropy_pytorch, logits, targets)
        diff = (fused_cross_entropy(logits, targets) - fused_cross_entropy_pytorch(logits, targets)).abs().max().item()
        record("fused_ce", "loss", t_tri, t_pt, diff)
    except Exception as e:
        print(f"fused_ce error: {e}")

    # 3. Cosformer
    try:
        from neuralops.ops.triton.cosformer_triton_kernel import triton_cosformer_linear_attention, triton_cosformer_linear_attention_pytorch_ref
        B, H, T, D = 2, 8, 512, 64
        q = torch.randn(B, H, T, D, device=device, dtype=dtype)
        k = torch.randn(B, H, T, D, device=device, dtype=dtype)
        v = torch.randn(B, H, T, D, device=device, dtype=dtype)
        t_tri = time_fn(triton_cosformer_linear_attention, q, k, v)
        t_pt = time_fn(triton_cosformer_linear_attention_pytorch_ref, q, k, v)
        diff = (triton_cosformer_linear_attention(q, k, v) - triton_cosformer_linear_attention_pytorch_ref(q, k, v)).abs().max().item()
        record("cosformer", "linear_attention", t_tri, t_pt, diff)
    except Exception as e:
        print(f"cosformer error: {e}")

    # 4. HGRN2 Scan
    try:
        from neuralops.ops.triton.hgrn2_scan import hgrn2_scan, hgrn2_scan_pytorch
        B, N, dim = 4, 512, 256
        x = torch.randn(B, N, dim, device=device, dtype=dtype)
        g = torch.randn(B, N, dim, device=device, dtype=dtype).sigmoid()
        t_tri = time_fn(hgrn2_scan, x, g)
        t_pt = time_fn(hgrn2_scan_pytorch, x, g)
        diff = (hgrn2_scan(x, g) - hgrn2_scan_pytorch(x, g)).abs().max().item()
        record("hgrn2_scan", "recurrent", t_tri, t_pt, diff)
    except Exception as e:
        print(f"hgrn2_scan error: {e}")

    # 5. GLA Scan
    try:
        from neuralops.ops.triton.gla_scan import gla_scan, gla_scan_pytorch
        B, H, N, D = 4, 8, 512, 64
        q = torch.randn(B, H, N, D, device=device, dtype=dtype)
        k = torch.randn(B, H, N, D, device=device, dtype=dtype)
        v = torch.randn(B, H, N, D, device=device, dtype=dtype)
        g = -torch.rand(B, H, N, D, device=device, dtype=dtype)
        t_tri = time_fn(gla_scan, q, k, v, g)
        t_pt = time_fn(gla_scan_pytorch, q, k, v, g)
        diff = (gla_scan(q, k, v, g) - gla_scan_pytorch(q, k, v, g)).abs().max().item()
        record("gla_scan", "recurrent", t_tri, t_pt, diff)
    except Exception as e:
        print(f"gla_scan error: {e}")

    # 6. Selective Scan
    try:
        from neuralops.ops.triton.selective_scan import selective_scan_fn, selective_scan_ref
        B, D, L = 4, 64, 512
        N = 16
        u = torch.randn(B, D, L, device=device, dtype=dtype)
        delta = torch.randn(B, D, L, device=device, dtype=dtype).abs() + 0.1
        A = -torch.rand(D, N, device=device, dtype=dtype) - 0.1
        B_mat = torch.randn(B, N, L, device=device, dtype=dtype)
        C_mat = torch.randn(B, N, L, device=device, dtype=dtype)
        D_vec = torch.randn(D, device=device, dtype=dtype)
        t_tri = time_fn(selective_scan_fn, u, delta, A, B_mat, C_mat, D_vec)
        t_pt = time_fn(selective_scan_ref, u, delta, A, B_mat, C_mat, D_vec)
        diff = (selective_scan_fn(u, delta, A, B_mat, C_mat, D_vec) - selective_scan_ref(u, delta, A, B_mat, C_mat, D_vec)).abs().max().item()
        record("selective_scan", "ssm", t_tri, t_pt, diff)
    except Exception as e:
        print(f"selective_scan error: {e}")

    # 7. Flash Decoding
    try:
        from neuralops.ops.triton.flash_decoding import flash_decode_kernel, flash_decoding_pytorch
        B, H, D, S = 4, 8, 64, 1024
        q = torch.randn(B, H, 1, D, device=device, dtype=dtype)
        k = torch.randn(B, H, S, D, device=device, dtype=dtype)
        v = torch.randn(B, H, S, D, device=device, dtype=dtype)
        t_tri = time_fn(flash_decode_kernel, q, k, v)
        t_pt = time_fn(flash_decoding_pytorch, q, k, v)
        diff = (flash_decode_kernel(q, k, v) - flash_decoding_pytorch(q, k, v)).abs().max().item()
        record("flash_decoding", "attention", t_tri, t_pt, diff)
    except Exception as e:
        print(f"flash_decoding error: {e}")

    # 8. Fused Goldfinch Hopfield
    try:
        from neuralops.ops.triton.fms_scan import fused_goldfinch_hopfield_lookup_triton, fused_goldfinch_hopfield_lookup_pytorch
        B, H, T, D = 4, 8, 512, 128
        chunk = 16
        q = torch.randn(B, H, T, D, device=device, dtype=dtype)
        mem = torch.randn(B, H, chunk, D, device=device, dtype=dtype)
        val = torch.randn(B, H, chunk, D, device=device, dtype=dtype)
        t_tri = time_fn(fused_goldfinch_hopfield_lookup_triton, q, mem, val)
        t_pt = time_fn(fused_goldfinch_hopfield_lookup_pytorch, q, mem, val)
        diff = (fused_goldfinch_hopfield_lookup_triton(q, mem, val) - fused_goldfinch_hopfield_lookup_pytorch(q, mem, val)).abs().max().item()
        record("fused_goldfinch_hopfield", "hopfield", t_tri, t_pt, diff)
    except Exception as e:
        print(f"fused_goldfinch_hopfield error: {e}")

    # 9. Fused Unified RoPE
    try:
        from neuralops.ops.triton.fused_rope import fused_unified_rope_offset_triton, fused_unified_rope_offset_pytorch
        B, H, T, D = 4, 8, 512, 64
        x = torch.randn(B, H, T, D, device=device, dtype=dtype)
        cos = torch.randn(1, 1, T, D, device=device, dtype=dtype)
        sin = torch.randn(1, 1, T, D, device=device, dtype=dtype)
        t_tri = time_fn(fused_unified_rope_offset_triton, x, cos, sin, 64)
        t_pt = time_fn(fused_unified_rope_offset_pytorch, x, cos, sin)
        diff = (fused_unified_rope_offset_triton(x, cos, sin, 64) - fused_unified_rope_offset_pytorch(x, cos, sin)).abs().max().item()
        record("fused_rope", "rotary", t_tri, t_pt, diff)
    except Exception as e:
        print(f"fused_rope error: {e}")

    # 10. Fused Norm Linear
    try:
        from neuralops.ops.triton.fused_norm_linear import fused_rmsnorm_gate_triton, fused_rmsnorm_gate
        B, N, dim = 4, 512, 2048
        x = torch.randn(B, N, dim, device=device, dtype=dtype)
        w = torch.randn(dim, device=device, dtype=dtype)
        g = torch.randn(B, N, dim, device=device, dtype=dtype)
        t_tri = time_fn(fused_rmsnorm_gate_triton, x, w, g)
        t_pt = time_fn(fused_rmsnorm_gate, x, w, g)
        diff = (fused_rmsnorm_gate_triton(x, w, g) - fused_rmsnorm_gate(x, w, g)).abs().max().item()
        record("fused_norm_linear", "normalization", t_tri, t_pt, diff)
    except Exception as e:
        print(f"fused_norm_linear error: {e}")

    # 11. Fused MQA Bias Scale
    try:
        from neuralops.ops.triton.gqa import fused_mqa_bias_scale_triton, fused_mqa_bias_scale_pytorch
        B, H, T, D = 4, 8, 256, 256
        q = torch.randn(B, H, T, D, device=device, dtype=dtype)
        bias = torch.randn(1, H, 1, 1, device=device, dtype=dtype)
        scale = 0.125
        t_tri = time_fn(fused_mqa_bias_scale_triton, q, bias, scale)
        t_pt = time_fn(fused_mqa_bias_scale_pytorch, q, bias, scale)
        diff = (fused_mqa_bias_scale_triton(q, bias, scale) - fused_mqa_bias_scale_pytorch(q, bias, scale)).abs().max().item()
        record("fused_mqa_bias_scale", "gqa", t_tri, t_pt, diff)
    except Exception as e:
        print(f"fused_mqa_bias_scale error: {e}")

    # 12. Fused Hamming Scores
    try:
        from neuralops.ops.triton.sparse_attn import fused_hamming_scores_triton, fused_hamming_scores_pytorch
        B, H, T, D = 4, 8, 256, 64
        codes_q = torch.randint(0, 256, (B, H, T), device=device, dtype=torch.uint8)
        codes_k = torch.randint(0, 256, (B, H, T), device=device, dtype=torch.uint8)
        t_tri = time_fn(fused_hamming_scores_triton, codes_q, codes_k, 1.0, 1.0, 1.0)
        t_pt = time_fn(fused_hamming_scores_pytorch, codes_q, codes_k, 1.0, 1.0, 1.0)
        diff = (fused_hamming_scores_triton(codes_q, codes_k, 1.0, 1.0, 1.0) - fused_hamming_scores_pytorch(codes_q, codes_k, 1.0, 1.0, 1.0)).abs().max().item()
        record("fused_hamming_scores", "sparse", t_tri, t_pt, diff)
    except Exception as e:
        print(f"fused_hamming_scores error: {e}")

    # 13. Fused Favor Plus
    try:
        from neuralops.ops.triton.elementwise import fused_favor_plus_feature_map_triton, fused_favor_plus_feature_map_pytorch
        B, H, N, D = 4, 8, 256, 64
        M = 128
        x = torch.randn(B, H, N, D, device=device, dtype=dtype)
        omega = torch.randn(M, D, device=device, dtype=dtype)
        t_tri = time_fn(fused_favor_plus_feature_map_triton, x, omega)
        t_pt = time_fn(fused_favor_plus_feature_map_pytorch, x, omega)
        diff = (fused_favor_plus_feature_map_triton(x, omega) - fused_favor_plus_feature_map_pytorch(x, omega)).abs().max().item()
        record("fused_favor_plus", "elementwise", t_tri, t_pt, diff)
    except Exception as e:
        print(f"fused_favor_plus error: {e}")

    # 14. Fused RALA Augmented Scan
    try:
        from neuralops.ops.triton.linear_attn import fused_rala_augmented_scan_triton, fused_rala_augmented_scan_pytorch
        T, D, r = 512, 32, 4
        q = torch.randn(T, D, device=device, dtype=dtype)
        k = torch.randn(T, D, device=device, dtype=dtype)
        v = torch.randn(T, D, device=device, dtype=dtype)
        u = torch.randn(T, r, device=device, dtype=dtype)
        b = torch.randn(T, r, device=device, dtype=dtype)
        t_tri = time_fn(fused_rala_augmented_scan_triton, q, k, v, u, b)
        t_pt = time_fn(fused_rala_augmented_scan_pytorch, q, k, v, u, b)
        diff = (fused_rala_augmented_scan_triton(q, k, v, u, b) - fused_rala_augmented_scan_pytorch(q, k, v, u, b)).abs().max().item()
        record("fused_rala_augmented_scan", "linear_attention", t_tri, t_pt, diff)
    except Exception as e:
        print(f"fused_rala_augmented_scan error: {e}")

    # 15. Fused Linformer Seq Proj
    try:
        from neuralops.ops.triton.low_rank import fused_linformer_seq_proj_triton, fused_linformer_seq_proj_pytorch
        B, H, N, D = 4, 8, 512, 64
        k = 128
        x = torch.randn(B, H, N, D, device=device, dtype=dtype)
        e = torch.randn(k, N, device=device, dtype=dtype)
        t_tri = time_fn(fused_linformer_seq_proj_triton, x, e)
        t_pt = time_fn(fused_linformer_seq_proj_pytorch, x, e)
        diff = (fused_linformer_seq_proj_triton(x, e) - fused_linformer_seq_proj_pytorch(x, e)).abs().max().item()
        record("fused_linformer_seq_proj", "low_rank", t_tri, t_pt, diff)
    except Exception as e:
        print(f"fused_linformer_seq_proj error: {e}")

    # 16. Fused HNC Nullspace Step
    try:
        from neuralops.ops.triton.low_rank import fused_hnc_nullspace_step_triton, fused_hnc_nullspace_step_pytorch
        dim = 2048
        W = torch.randn(dim, dim, device=device, dtype=dtype)
        null_basis = torch.randn(dim, 64, device=device, dtype=dtype)
        t_tri = time_fn(fused_hnc_nullspace_step_triton, W, null_basis)
        t_pt = time_fn(fused_hnc_nullspace_step_pytorch, W, null_basis)
        diff = (fused_hnc_nullspace_step_triton(W, null_basis) - fused_hnc_nullspace_step_pytorch(W, null_basis)).abs().max().item()
        record("fused_hnc_nullspace_step", "low_rank", t_tri, t_pt, diff)
    except Exception as e:
        print(f"fused_hnc_nullspace_step error: {e}")

    # 17. Fused KD Loss
    try:
        from neuralops.ops.triton.fused_kd import fused_kd_loss, fused_kd_loss_pytorch
        BN, V = 2048, 16384
        s_logits = torch.randn(BN, V, device=device, dtype=dtype)
        t_logits = torch.randn(BN, V, device=device, dtype=dtype)
        t_tri = time_fn(fused_kd_loss, s_logits, t_logits, 2.0, 0.5)
        t_pt = time_fn(fused_kd_loss_pytorch, s_logits, t_logits, 2.0, 0.5)
        diff = (fused_kd_loss(s_logits, t_logits, 2.0, 0.5) - fused_kd_loss_pytorch(s_logits, t_logits, 2.0, 0.5)).abs().max().item()
        record("fused_kd", "loss", t_tri, t_pt, diff)
    except Exception as e:
        print(f"fused_kd error: {e}")

    # 18. FMS Scan
    try:
        from neuralops.ops.triton.fms_scan import fms_scan, fms_scan_pytorch
        B, H, N, D = 4, 8, 512, 64
        q = torch.randn(B, H, N, D, device=device, dtype=dtype)
        k = torch.randn(B, H, N, D, device=device, dtype=dtype)
        v = torch.randn(B, H, N, D, device=device, dtype=dtype)
        decay = torch.sigmoid(torch.randn(H, device=device, dtype=dtype))
        t_tri = time_fn(fms_scan, q, k, v, decay)
        t_pt = time_fn(fms_scan_pytorch, q, k, v, decay)
        diff = (fms_scan(q, k, v, decay) - fms_scan_pytorch(q, k, v, decay)).abs().max().item()
        record("fms_scan", "state_space", t_tri, t_pt, diff)
    except Exception as e:
        print(f"fms_scan error: {e}")

    # 19. Based Scan
    try:
        from neuralops.ops.triton.based_scan import based_taylor_scan, based_taylor_scan_pytorch
        B, H, T, D = 2, 4, 256, 64
        q = torch.randn(B, H, T, D, device=device, dtype=dtype)
        k = torch.randn(B, H, T, D, device=device, dtype=dtype)
        v = torch.randn(B, H, T, D, device=device, dtype=dtype)
        t_tri = time_fn(based_taylor_scan, q, k, v)
        t_pt = time_fn(based_taylor_scan_pytorch, q, k, v)
        diff = (based_taylor_scan(q, k, v) - based_taylor_scan_pytorch(q, k, v)).abs().max().item()
        record("based_scan", "linear_attention", t_tri, t_pt, diff)
    except Exception as e:
        print(f"based_scan error: {e}")

    # 20. RWKV6 Scan
    try:
        from neuralops.ops.triton.rwkv6_scan import rwkv6_scan, rwkv6_scan_pytorch
        B, H, T, D = 2, 4, 256, 64
        r = torch.randn(B, H, T, D, device=device, dtype=dtype)
        k = torch.randn(B, H, T, D, device=device, dtype=dtype)
        v = torch.randn(B, H, T, D, device=device, dtype=dtype)
        w = -torch.rand(B, H, T, D, device=device, dtype=dtype)
        u = torch.randn(H, D, device=device, dtype=dtype)
        t_tri = time_fn(rwkv6_scan, r, k, v, w, u)
        t_pt = time_fn(rwkv6_scan_pytorch, r, k, v, w, u)
        diff = (rwkv6_scan(r, k, v, w, u) - rwkv6_scan_pytorch(r, k, v, w, u)).abs().max().item()
        record("rwkv6_scan", "recurrent", t_tri, t_pt, diff)
    except Exception as e:
        print(f"rwkv6_scan error: {e}")

    # Save to CSV
    os.makedirs("results/reruns_2026-11-03", exist_ok=True)
    out_csv = "results/reruns_2026-11-03/all_triton_modules_comprehensive.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["module", "category", "triton_ms", "pytorch_ms", "speedup", "max_diff", "status"])
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"\nSaved all Triton module benchmark results to: {out_csv}")

if __name__ == "__main__":
    run_benchmarks()
