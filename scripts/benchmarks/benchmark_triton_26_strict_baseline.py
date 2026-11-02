#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Rigorous Physical Silicon Baseline Benchmark for 26 Canonical Triton Kernels.

Engineering Standards:
- Fail-Closed: Immediate RuntimeError if CUDA is unavailable.
- Timing: High-precision torch.cuda.Event with synchronize before and after.
- Cache Management: 256MB L2 cache eviction tensor before each timed run.
- Stability: 10 warmup iterations.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F

# Fail-Closed Guard
if not torch.cuda.is_available():
    raise RuntimeError(
        "FAIL-CLOSED: CUDA is not available! This benchmark requires physical GPU execution. "
        "CPU fallback is strictly prohibited."
    )


class CacheFlusher:
    """Evicts L2 GPU cache by writing to a dedicated 256MB memory buffer."""

    def __init__(self, size_mb: int = 256, device: torch.device | None = None) -> None:
        self.device = device or torch.device("cuda:0")
        num_floats = (size_mb * 1024 * 1024) // 4
        self.buffer = torch.empty(num_floats, dtype=torch.float32, device=self.device)

    def flush(self) -> None:
        self.buffer.zero_()
        torch.cuda.synchronize(self.device)


def benchmark_kernel_pair(
    name: str,
    shape_desc: str,
    triton_fn,
    pytorch_fn,
    args: tuple,
    kwargs: dict,
    flusher: CacheFlusher,
    warmup: int = 10,
    repeat: int = 30,
) -> dict:
    device = torch.device("cuda:0")

    # 1. Warmup (triggers JIT compilation and autotuner)
    for _ in range(warmup):
        _ = triton_fn(*args, **kwargs)
        _ = pytorch_fn(*args, **kwargs)
    torch.cuda.synchronize(device)

    # 2. Benchmark Triton Kernel with L2 Cache Flushing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    triton_times_ms: list[float] = []
    for _ in range(repeat):
        flusher.flush()
        torch.cuda.synchronize(device)
        start_event.record()
        out_tri = triton_fn(*args, **kwargs)
        end_event.record()
        torch.cuda.synchronize(device)
        triton_times_ms.append(start_event.elapsed_time(end_event))

    avg_triton_ms = sum(triton_times_ms) / len(triton_times_ms)

    # 3. Benchmark PyTorch Reference with L2 Cache Flushing
    pytorch_times_ms: list[float] = []
    for _ in range(repeat):
        flusher.flush()
        torch.cuda.synchronize(device)
        start_event.record()
        out_pt = pytorch_fn(*args, **kwargs)
        end_event.record()
        torch.cuda.synchronize(device)
        pytorch_times_ms.append(start_event.elapsed_time(end_event))

    avg_pytorch_ms = sum(pytorch_times_ms) / len(pytorch_times_ms)

    # 4. Numerical Parity Verification
    if isinstance(out_tri, tuple):
        out_tri = out_tri[0]
    if isinstance(out_pt, tuple):
        out_pt = out_pt[0]

    has_nonfinite = not bool(torch.isfinite(out_tri).all().item()) or not bool(torch.isfinite(out_pt).all().item())
    max_diff = (out_tri.float() - out_pt.float()).abs().max().item()
    tolerance = 1e-3 if out_pt.dtype == torch.float16 else 1e-4

    speedup = round(avg_pytorch_ms / max(avg_triton_ms, 1e-6), 2)
    if has_nonfinite:
        status = "FAIL_NAN"
    elif max_diff > tolerance:
        status = "FAIL_NUMERICS"
    else:
        status = "OK"

    return {
        "kernel": name,
        "shape": shape_desc,
        "triton_ms": round(avg_triton_ms, 3),
        "pytorch_ms": round(avg_pytorch_ms, 3),
        "speedup": speedup,
        "max_diff": round(max_diff, 6),
        "status": status,
    }


def run_benchmark_suite(output_csv: str, output_json: str) -> list[dict]:
    device = torch.device("cuda:0")
    dtype = torch.float16
    flusher = CacheFlusher(size_mb=256, device=device)

    results = []

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Precision: {dtype}")
    print("-" * 88)
    print(f"{'Kernel Name':<32} | {'Shape':<18} | {'PyTorch':<10} | {'Triton':<10} | {'Speedup':<8} | {'Status'}")
    print("-" * 88)

    def eval_op(name, shape, tri_fn, pt_fn, *args, **kwargs):
        try:
            res = benchmark_kernel_pair(name, shape, tri_fn, pt_fn, args, kwargs, flusher)
            results.append(res)
            print(f"{name:<32} | {shape:<18} | {res['pytorch_ms']:>7.3f} ms | {res['triton_ms']:>7.3f} ms | {res['speedup']:>6.2f}x | {res['status']}")
        except Exception as exc:
            err_res = {
                "kernel": name,
                "shape": shape,
                "triton_ms": 0.0,
                "pytorch_ms": 0.0,
                "speedup": 0.0,
                "max_diff": float("inf"),
                "status": f"ERROR: {exc}",
            }
            results.append(err_res)
            print(f"{name:<32} | {shape:<18} | {'ERROR':<10} | {'ERROR':<10} | {'0.00x':<8} | {exc}")

    # 1. fused_hnc_nullspace_step
    from neuralops.core.triton_kernels.low_rank import fused_hnc_nullspace_step_pytorch, fused_hnc_nullspace_step_triton
    w = torch.randn(2048, 2048, device=device, dtype=dtype)
    g = torch.randn(2048, 2048, device=device, dtype=dtype)
    r = torch.randn(2048, 2048, device=device, dtype=dtype)
    eval_op("fused_hnc_nullspace_step", "(2048x2048)", fused_hnc_nullspace_step_triton, fused_hnc_nullspace_step_pytorch, w, g, r, 0.05)

    # 2. fused_linformer_seq_proj
    from neuralops.core.triton_kernels.low_rank import fused_linformer_seq_proj_pytorch, fused_linformer_seq_proj_triton
    x_lin = torch.randn(8, 16, 4096, 64, device=device, dtype=dtype)
    p_lin = torch.randn(64, 4096, device=device, dtype=dtype)
    eval_op("fused_linformer_seq_proj", "(T=4096)", fused_linformer_seq_proj_triton, fused_linformer_seq_proj_pytorch, p_lin, x_lin)

    # 3. fused_kronecker_gemm
    from neuralops.core.triton_kernels.low_rank import fused_kronecker_batched_gemm_pytorch, fused_kronecker_batched_gemm_triton
    x_k = torch.randn(4, 128, 256, device=device, dtype=dtype)
    A_k = torch.randn(16, 16, device=device, dtype=dtype)
    B_k = torch.randn(16, 16, device=device, dtype=dtype)
    eval_op("fused_kronecker_gemm", "(dim=256, r=16)", fused_kronecker_batched_gemm_triton, fused_kronecker_batched_gemm_pytorch, x_k, A_k, B_k)

    # 4. fused_nystrom_landmark_pinv
    from neuralops.core.triton_kernels.low_rank import fused_nystrom_landmark_pinv_step_pytorch, fused_nystrom_landmark_pinv_step_triton
    m_nys, N_nys = 32, 512
    F1 = F.softmax(torch.randn(4, 8, N_nys, m_nys, device=device, dtype=torch.float32), dim=-1)
    F2_pinv = torch.randn(4, 8, m_nys, m_nys, device=device, dtype=torch.float32)
    F3 = F.softmax(torch.randn(4, 8, m_nys, N_nys, device=device, dtype=torch.float32), dim=-1)
    V_nys = torch.randn(4, 8, N_nys, 64, device=device, dtype=torch.float32)
    eval_op("fused_nystrom_landmark_pinv", f"(m={m_nys}, N={N_nys})", fused_nystrom_landmark_pinv_step_triton, fused_nystrom_landmark_pinv_step_pytorch, F1, F2_pinv, F3, V_nys)

    # 5. sliding_window_attention
    from neuralops.core.triton_kernels.sliding_window import sliding_window_attention, sliding_window_pytorch
    seq_len = 2048
    q_sw = torch.randn(2, 4, seq_len, 64, device=device, dtype=dtype)
    k_sw = torch.randn(2, 4, seq_len, 64, device=device, dtype=dtype)
    v_sw = torch.randn(2, 4, seq_len, 64, device=device, dtype=dtype)
    eval_op("sliding_window_attention", f"(T={seq_len}, W=128)", sliding_window_attention, sliding_window_pytorch, q_sw, k_sw, v_sw, 128)

    # 6. gla_scan
    from neuralops.core.triton_kernels.gla_scan import gla_scan, gla_scan_pytorch
    seq_len = 1024
    q_gla = F.elu(torch.randn(2, 4, seq_len, 64, device=device, dtype=dtype)) + 1.0
    k_gla = F.elu(torch.randn(2, 4, seq_len, 64, device=device, dtype=dtype)) + 1.0
    v_gla = torch.randn(2, 4, seq_len, 64, device=device, dtype=dtype)
    g_gla = torch.sigmoid(torch.randn(2, 4, seq_len, device=device, dtype=dtype)).clamp(1e-4, 1.0 - 1e-4)
    eval_op("gla_scan", f"(T={seq_len})", gla_scan, gla_scan_pytorch, q_gla, k_gla, v_gla, g_gla)

    # 7. gdn_scan
    from neuralops.core.triton_kernels.gdn_scan import gdn_scan, gdn_scan_pytorch
    seq_len = 1024
    q_gdn = torch.randn(2, 4, seq_len, 64, device=device, dtype=dtype)
    k_gdn = F.normalize(torch.randn(2, 4, seq_len, 64, device=device, dtype=dtype), dim=-1)
    v_gdn = torch.randn(2, 4, seq_len, 64, device=device, dtype=dtype)
    beta_gdn = torch.sigmoid(torch.randn(2, 4, seq_len, device=device, dtype=dtype))
    alpha_gdn = torch.sigmoid(torch.randn(2, 4, seq_len, device=device, dtype=dtype))
    eval_op("gdn_scan", f"(T={seq_len})", gdn_scan, gdn_scan_pytorch, q_gdn, k_gdn, v_gdn, beta_gdn, alpha_gdn)

    # 8. hgrn2_scan
    from neuralops.core.triton_kernels.hgrn2_scan import hgrn2_scan, hgrn2_scan_pytorch
    seq_len = 1024
    f_hgrn = torch.sigmoid(torch.randn(2, seq_len, 256, device=device, dtype=dtype))
    iv_hgrn = torch.randn(2, seq_len, 256, device=device, dtype=dtype)
    eval_op("hgrn2_scan", f"(T={seq_len})", lambda f, iv: hgrn2_scan(f, iv, res_clip=1.0), lambda f, iv: hgrn2_scan_pytorch(f, iv, res_clip=1.0), f_hgrn, iv_hgrn)

    # 9. softplus
    from neuralops.core.triton_kernels.softplus import softplus_fwd, softplus_pytorch
    num_elements = 4194304
    x_sp = torch.randn(num_elements, device=device, dtype=dtype)
    eval_op("softplus", f"({num_elements})", softplus_fwd, softplus_pytorch, x_sp)

    # 10. fused_norm_linear
    from neuralops.core.triton_kernels.fused_norm_linear import fused_rmsnorm_gate_triton, fused_rmsnorm_gate
    dim = 2048
    x_fnl = torch.randn(2, 512, dim, device=device, dtype=dtype)
    g_fnl = torch.randn(2, 512, dim, device=device, dtype=dtype)
    w_fnl = torch.randn(dim, device=device, dtype=dtype)
    eval_op("fused_norm_linear", f"({dim})", fused_rmsnorm_gate_triton, fused_rmsnorm_gate, x_fnl, g_fnl, w_fnl)

    # 11. cosformer
    from neuralops.core.triton_kernels.cosformer_triton_kernel import cosformer_pytorch_fwd, cosformer_triton_fwd
    seq_len = 1024
    q_cos = torch.randn(2, 4, seq_len, 64, device=device, dtype=dtype)
    k_cos = torch.randn(2, 4, seq_len, 64, device=device, dtype=dtype)
    v_cos = torch.randn(2, 4, seq_len, 64, device=device, dtype=dtype)
    scale_cos = 64.0 ** -0.5
    q_cos_act = F.elu(q_cos.float() * scale_cos) + 1.0
    k_cos_act = F.elu(k_cos.float()) + 1.0
    eval_op("cosformer", f"(T={seq_len})", lambda q, k, v: cosformer_triton_fwd(q, k, v, is_causal=False), lambda q, k, v: cosformer_pytorch_fwd(q, k, v, is_causal=False), q_cos_act, k_cos_act, v_cos.float())

    # 12. fused_ce
    from neuralops.core.triton_kernels.fused_ce import fused_cross_entropy, fused_cross_entropy_pytorch
    vocab = 50257
    logits_ce = torch.randn(8, 256, vocab, device=device, dtype=dtype)
    targets_ce = torch.randint(0, vocab, (8, 256), device=device)
    eval_op("fused_ce", f"(V={vocab})", fused_cross_entropy, fused_cross_entropy_pytorch, logits_ce, targets_ce)

    # 13. fused_kd
    from neuralops.core.triton_kernels.fused_kd import fused_kd_loss, fused_kd_loss_pytorch
    vocab = 16384
    s_logits = torch.randn(2, 64, vocab, device=device, dtype=dtype)
    t_logits = torch.randn(2, 64, vocab, device=device, dtype=dtype)
    eval_op("fused_kd", f"(V={vocab})", fused_kd_loss, fused_kd_loss_pytorch, s_logits, t_logits, 2.0)

    # 14. selective_scan
    from neuralops.core.triton_kernels.selective_scan import selective_scan_fn, selective_scan_ref
    seq_len = 512
    u_ssm = torch.randn(2, 64, seq_len, device=device, dtype=dtype)
    delta_ssm = F.softplus(torch.randn(2, 64, seq_len, device=device, dtype=dtype)) * 0.05
    A_ssm = -torch.rand(64, 16, device=device, dtype=dtype) - 0.1
    B_ssm = torch.randn(2, 16, seq_len, device=device, dtype=dtype) * 0.1
    C_ssm = torch.randn(2, 16, seq_len, device=device, dtype=dtype) * 0.1
    eval_op("selective_scan", f"(T={seq_len})", selective_scan_fn, selective_scan_ref, u_ssm, delta_ssm, A_ssm, B_ssm, C_ssm)

    # 15. fms_scan
    from neuralops.core.triton_kernels.fms_scan import fms_scan, fms_scan_pytorch
    seq_len = 1024
    q_fms = torch.randn(2, 4, seq_len, 64, device=device, dtype=dtype)
    k_fms = torch.randn(2, 4, seq_len, 64, device=device, dtype=dtype)
    v_fms = torch.randn(2, 4, seq_len, 64, device=device, dtype=dtype)
    decay_fms = torch.sigmoid(torch.randn(4, device=device, dtype=dtype))
    eval_op("fms_scan", f"(T={seq_len})", fms_scan, fms_scan_pytorch, q_fms, k_fms, v_fms, decay_fms)

    # 16. paged_attention
    from neuralops.core.triton_kernels.paged_attention import paged_attention_kernel, paged_attention_pytorch
    num_blocks, block_size, num_heads, head_dim = 128, 16, 4, 64
    k_cache = torch.randn(num_blocks, block_size, num_heads, head_dim, device=device, dtype=dtype)
    v_cache = torch.randn(num_blocks, block_size, num_heads, head_dim, device=device, dtype=dtype)
    block_tables = torch.arange(num_blocks, device=device, dtype=torch.int32).view(2, 64)
    context_lens = torch.full((2,), 64 * block_size, device=device, dtype=torch.int32)
    q_pa = torch.randn(2, num_heads, head_dim, device=device, dtype=dtype)
    eval_op("paged_attention", "(B=2, H=4, d=64)", paged_attention_kernel, paged_attention_pytorch, q_pa, k_cache, v_cache, block_tables, context_lens, block_size=block_size)

    # 17. fused_mqa_bias_scale
    from neuralops.core.triton_kernels.gqa import fused_mqa_bias_scale_triton, fused_mqa_bias_scale_pytorch
    scores_gqa = torch.randn(4, 8, 256, 256, device=device, dtype=torch.float32)
    eval_op("fused_mqa_bias_scale", "(256x256, causal)", fused_mqa_bias_scale_triton, fused_mqa_bias_scale_pytorch, scores_gqa, 0.25, is_causal=True, past_len=4)

    # 18. fused_hamming_scores
    from neuralops.core.triton_kernels.sparse_attn import fused_hamming_scores_triton, fused_hamming_scores_pytorch
    Q_ham = torch.randn(4, 8, 256, 64, device=device, dtype=torch.float32)
    K_ham = torch.randn(4, 8, 256, 64, device=device, dtype=torch.float32)
    lora_q = torch.randn(8, device=device, dtype=torch.float32)
    lora_k = torch.randn(8, device=device, dtype=torch.float32)
    temp = torch.tensor([1.2], device=device, dtype=torch.float32)
    eval_op("fused_hamming_scores", "(256x64)", fused_hamming_scores_triton, fused_hamming_scores_pytorch, Q_ham, K_ham, lora_q, lora_k, temp)

    # 19. fused_favor_plus
    from neuralops.core.triton_kernels.elementwise import fused_favor_plus_feature_map_triton, fused_favor_plus_feature_map_pytorch
    x_fav = torch.randn(4, 8, 256, 64, device=device, dtype=torch.float32)
    omega_fav = torch.randn(128, 64, device=device, dtype=torch.float32)
    eval_op("fused_favor_plus", "(256x64, M=128)", fused_favor_plus_feature_map_triton, fused_favor_plus_feature_map_pytorch, x_fav, omega_fav, num_features=128, kernel_type="favor_plus")

    # 20. fused_rope
    from neuralops.core.triton_kernels.fused_rope import fused_rope_fwd_kernel, apply_rope_rotation
    import triton
    B_r, H_r, N_r, D_r = 4, 8, 1024, 64
    q_r = torch.randn(B_r, H_r, N_r, D_r, device=device, dtype=dtype)
    k_r = torch.randn(B_r, H_r, N_r, D_r, device=device, dtype=dtype)
    cos_r = torch.cos(torch.randn(N_r, D_r // 2, device=device, dtype=dtype))
    sin_r = torch.sin(torch.randn(N_r, D_r // 2, device=device, dtype=dtype))

    def tri_rope_bench(q, k, cos, sin):
        out_q = torch.empty_like(q)
        out_k = torch.empty_like(k)
        BLOCK_M = 32
        grid = (triton.cdiv(N_r, BLOCK_M), B_r * H_r)
        fused_rope_fwd_kernel[grid](
            q, k, cos, sin, out_q, out_k,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            0, cos.stride(0), cos.stride(1),
            0, sin.stride(0), sin.stride(1),
            out_q.stride(0), out_q.stride(1), out_q.stride(2), out_q.stride(3),
            out_k.stride(0), out_k.stride(1), out_k.stride(2), out_k.stride(3),
            H_r, N_r, D_r, D_r // 2,
            BLOCK_M=BLOCK_M,
        )
        return out_q

    eval_op("fused_rope", f"(T={N_r}, d={D_r})", tri_rope_bench, lambda q, k, c, s: apply_rope_rotation(q, c, s), q_r, k_r, cos_r, sin_r)

    # 21. fused_moba_block_router
    from neuralops.core.triton_kernels.block_sparse import fused_moba_block_router_combine_triton, fused_moba_block_router_combine_pytorch
    Qb_moba = torch.randn(4, 4, 16, 16, 32, device=device, dtype=torch.float32)
    Kb_moba = torch.randn(4, 4, 16, 16, 32, device=device, dtype=torch.float32)
    Vb_moba = torch.randn(4, 4, 16, 16, 32, device=device, dtype=torch.float32)
    eval_op("fused_moba_block_router", "(16x16x32)", fused_moba_block_router_combine_triton, fused_moba_block_router_combine_pytorch, Qb_moba, Kb_moba, Vb_moba, 0.25, top_k=2, is_causal=True)

    # 22. fused_rala_augmented_scan
    from neuralops.core.triton_kernels.linear_attn import fused_rala_augmented_scan_triton, fused_rala_augmented_scan_pytorch
    q_rala = F.softplus(torch.randn(2, 4, 512, 32, device=device, dtype=dtype))
    k_rala = F.softplus(torch.randn(2, 4, 512, 32, device=device, dtype=dtype))
    v_rala = torch.randn(2, 4, 512, 32, device=device, dtype=dtype)
    lora_U = torch.randn(4, 32, 4, device=device, dtype=dtype)
    lora_V = torch.randn(4, 32, 4, device=device, dtype=dtype)
    eval_op("fused_rala_augmented_scan", "(512x32, r=4)", fused_rala_augmented_scan_triton, fused_rala_augmented_scan_pytorch, q_rala, k_rala, v_rala, lora_U, lora_V, is_causal=True)

    # 23. fused_recurrent_linear_step
    from neuralops.core.triton_kernels.state_update import fused_recurrent_linear_step
    q_rec = torch.randn(2, 4, 64, device=device, dtype=dtype)
    k_rec = torch.randn(2, 4, 64, device=device, dtype=dtype)
    v_rec = torch.randn(2, 4, 64, device=device, dtype=dtype)
    kv_state_rec = torch.randn(2, 4, 64, 64, device=device, dtype=dtype)
    z_state_rec = torch.randn(2, 4, 64, device=device, dtype=dtype)

    def py_rec_step(q_t, k_t, v_t, kv_s, z_s, eps=1e-6):
        kv = kv_s + torch.einsum("bhk,bhv->bhkv", k_t.float(), v_t.float()).to(kv_s.dtype)
        z = z_s + k_t
        num = torch.einsum("bhk,bhkv->bhv", q_t.float(), kv.float())
        den = torch.einsum("bhk,bhk->bh", q_t.float(), z.float()).unsqueeze(-1) + eps
        return (num / den).to(q_t.dtype), kv, z

    eval_op(
        "fused_recurrent_linear_step",
        "(B=2, H=4, D=64)",
        lambda q, k, v, kv, z: fused_recurrent_linear_step(q, k, v, kv.clone(), z.clone()),
        lambda q, k, v, kv, z: py_rec_step(q, k, v, kv.clone(), z.clone()),
        q_rec, k_rec, v_rec, kv_state_rec, z_state_rec,
    )

    # 24. fused_jamba_hybrid_gate
    from neuralops.core.triton_kernels.selective_scan import fused_jamba_hybrid_ssm_attn_gate_triton, fused_jamba_hybrid_ssm_attn_gate_pytorch
    attn_out_j = torch.randn(8, 512, 1024, device=device, dtype=torch.float32)
    ssm_out_j = torch.randn(8, 512, 1024, device=device, dtype=torch.float32)
    alpha_j = torch.randn(16, device=device, dtype=torch.float32)
    eval_op("fused_jamba_hybrid_gate", "(512x1024, H=16)", fused_jamba_hybrid_ssm_attn_gate_triton, fused_jamba_hybrid_ssm_attn_gate_pytorch, attn_out_j, ssm_out_j, alpha_j)

    # 25. fused_goldfinch_hopfield
    from neuralops.core.triton_kernels.fms_scan import fused_goldfinch_hopfield_lookup_triton, fused_goldfinch_hopfield_lookup_pytorch
    g_gf = torch.sigmoid(torch.randn(4, 512, 128, device=device, dtype=torch.float32))
    v_gf = torch.tanh(torch.randn(4, 512, 128, device=device, dtype=torch.float32))
    eval_op("fused_goldfinch_hopfield", "(512x128, chunk=16)", fused_goldfinch_hopfield_lookup_triton, fused_goldfinch_hopfield_lookup_pytorch, g_gf, v_gf, chunk_size=16)

    # 26. flash_decode
    from neuralops.core.triton_kernels.flash_decoding import flash_decode_kernel, flash_decoding_pytorch
    q_fd = torch.randn(2, 4, 1, 64, device=device, dtype=dtype)
    k_fd = torch.randn(2, 4, 1024, 64, device=device, dtype=dtype)
    v_fd = torch.randn(2, 4, 1024, 64, device=device, dtype=dtype)
    eval_op("flash_decode", "(N_kv=1024, H=4)", flash_decode_kernel, flash_decoding_pytorch, q_fd, k_fd, v_fd, chunk_size=64)

    # Save CSV and JSON
    if output_csv:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            wtr = csv.DictWriter(f, fieldnames=["kernel", "shape", "triton_ms", "pytorch_ms", "speedup", "max_diff", "status"])
            wtr.writeheader()
            for r in results:
                wtr.writerow(r)

    if output_json:
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description="Strict 26 Triton Kernel Baseline Benchmark on Physical GPU")
    parser.add_argument("--output-csv", default="results/triton_26_kernels_rtx3090_baseline.csv", help="Path to output CSV")
    parser.add_argument("--output-json", default="results/triton_26_kernels_rtx3090_baseline.json", help="Path to output JSON")
    args = parser.parse_args()

    run_benchmark_suite(args.output_csv, args.output_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
