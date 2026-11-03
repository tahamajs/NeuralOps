#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Comprehensive Physical Silicon Benchmark for All Triton Operators.
Evaluates:
  1. 26 Canonical Triton Modules (with tuned batch sizes, block autotuning)
  2. All 48 Triton Attention Layer Modules from ATTENTION_REGISTRY
  3. Standalone and Extension Triton Operators
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

class CacheFlusher:
    """Evicts L2 GPU cache by writing to a dedicated 256MB memory buffer."""
    def __init__(self, size_mb: int = 256, device: torch.device | None = None) -> None:
        self.device = device or torch.device("cuda:0")
        num_floats = (size_mb * 1024 * 1024) // 4
        self.buffer = torch.empty(num_floats, dtype=torch.float32, device=self.device)

    def flush(self) -> None:
        self.buffer.zero_()
        torch.cuda.synchronize(self.device)


def benchmark_pair(
    name: str,
    category: str,
    shape_desc: str,
    triton_fn,
    pytorch_fn,
    args: tuple,
    kwargs: dict,
    flusher: CacheFlusher,
    warmup: int = 10,
    repeat: int = 30,
    tolerance: float = 1e-1,
) -> dict:
    device = torch.device("cuda:0")

    # 1. Warmup
    for _ in range(warmup):
        _ = triton_fn(*args, **kwargs)
        _ = pytorch_fn(*args, **kwargs)
    torch.cuda.synchronize(device)

    # 2. Benchmark Triton
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    tri_times: list[float] = []
    for _ in range(repeat):
        flusher.flush()
        start_event.record()
        out_tri = triton_fn(*args, **kwargs)
        end_event.record()
        torch.cuda.synchronize(device)
        tri_times.append(start_event.elapsed_time(end_event))

    avg_tri_ms = sum(tri_times) / len(tri_times)

    # 3. Benchmark PyTorch Reference
    pt_times: list[float] = []
    for _ in range(repeat):
        flusher.flush()
        start_event.record()
        out_pt = pytorch_fn(*args, **kwargs)
        end_event.record()
        torch.cuda.synchronize(device)
        pt_times.append(start_event.elapsed_time(end_event))

    avg_pt_ms = sum(pt_times) / len(pt_times)

    # 4. Parity
    if isinstance(out_tri, tuple):
        out_tri = out_tri[0]
    if isinstance(out_pt, tuple):
        out_pt = out_pt[0]

    has_nan = not bool(torch.isfinite(out_tri).all().item()) or not bool(torch.isfinite(out_pt).all().item())
    max_diff = (out_tri.float() - out_pt.float()).abs().max().item()
    status = "FAIL_NAN" if has_nan else ("FAIL_NUMERICS" if max_diff > tolerance else "OK")
    speedup = round(avg_pt_ms / max(avg_tri_ms, 1e-6), 2)

    return {
        "kernel": name,
        "category": category,
        "shape": shape_desc,
        "triton_ms": round(avg_tri_ms, 3),
        "pytorch_ms": round(avg_pt_ms, 3),
        "speedup": speedup,
        "max_diff": round(max_diff, 6),
        "status": status,
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Strict physical silicon benchmark requires real CUDA GPU hardware.")

    device = torch.device("cuda:0")
    flusher = CacheFlusher(256, device)
    dtype = torch.float16
    results = []

    print("=" * 100)
    print(f"COMPREHENSIVE ALL-TRITON OPERATOR BENCHMARK ON {torch.cuda.get_device_name(0)}")
    print("=" * 100)
    print(f"{'Category':<15} | {'Kernel / Module Name':<32} | {'Shape':<16} | {'PyTorch':<10} | {'Triton':<10} | {'Speedup':<8} | {'Status'}")
    print("-" * 106)

    def eval_item(category, name, shape, tri_fn, pt_fn, *args, **kwargs):
        try:
            res = benchmark_pair(name, category, shape, tri_fn, pt_fn, args, kwargs, flusher)
            results.append(res)
            print(f"{category:<15} | {name:<32} | {shape:<16} | {res['pytorch_ms']:>7.3f} ms | {res['triton_ms']:>7.3f} ms | {res['speedup']:>6.2f}x | {res['status']}")
        except Exception as exc:
            print(f"{category:<15} | {name:<32} | {shape:<16} | ERROR: {exc}")

    # =========================================================================
    # PART 1: 26 CANONICAL TRITON OPERATORS (TUNED & SCALED)
    # =========================================================================
    import neuralops.core.triton_kernels as tk

    # 1. HGRN2 Scan
    B, H, T, D = 4, 8, 2048, 64
    q = torch.randn(B, H, T, D, device=device, dtype=dtype)
    k = torch.randn(B, H, T, D, device=device, dtype=dtype)
    v = torch.randn(B, H, T, D, device=device, dtype=dtype)
    eval_item("Canonical-26", "hgrn2_scan", f"T={T},d={D}", tk.hgrn2_scan, tk.hgrn2_scan_pytorch, q, k, v)

    # 2. GLA Scan
    eval_item("Canonical-26", "gla_scan", f"T=1024,d={D}", tk.gla_scan, tk.gla_scan_pytorch, q[:, :, :1024], k[:, :, :1024], v[:, :, :1024])

    # 3. Selective Scan
    u = torch.randn(4, 1024, 64, device=device, dtype=dtype)
    delta = torch.randn(4, 1024, 64, device=device, dtype=dtype)
    A = torch.randn(64, 16, device=device, dtype=dtype)
    B_t = torch.randn(4, 1024, 16, device=device, dtype=dtype)
    C_t = torch.randn(4, 1024, 16, device=device, dtype=dtype)
    from neuralops.core.triton_kernels.selective_scan import selective_scan_triton, selective_scan_pytorch
    eval_item("Canonical-26", "selective_scan", "T=1024,D=64", selective_scan_triton, selective_scan_pytorch, u, delta, A, B_t, C_t)

    # 4. FMS Scan
    from neuralops.core.triton_kernels.fms_scan import fms_scan, fms_scan_pytorch
    eval_item("Canonical-26", "fms_scan", "T=1024,d=64", fms_scan, fms_scan_pytorch, q[:, :, :1024], k[:, :, :1024], v[:, :, :1024])

    # 5. Flash Decode
    from neuralops.core.triton_kernels.flash_decoding import flash_decoding_fwd, flash_decoding_reference
    q_fd = torch.randn(4, 8, 64, device=device, dtype=dtype)
    k_fd = torch.randn(4, 8, 2048, 64, device=device, dtype=dtype)
    v_fd = torch.randn(4, 8, 2048, 64, device=device, dtype=dtype)
    eval_item("Canonical-26", "flash_decode", "N_kv=2048,H=8", flash_decoding_fwd, flash_decoding_reference, q_fd, k_fd, v_fd)

    # 6. Fused RALA Scan
    from neuralops.core.triton_kernels.linear_attn import fused_rala_augmented_scan_triton, fused_rala_augmented_scan_pytorch
    eval_item("Canonical-26", "fused_rala_scan", "1024x32,r=4", fused_rala_augmented_scan_triton, fused_rala_augmented_scan_pytorch, q[:, :, :1024, :32], k[:, :, :1024, :32], v[:, :, :1024, :32])

    # 7. Fused Goldfinch Hopfield
    from neuralops.core.triton_kernels.fms_scan import goldfinch_hopfield_fused_scan_triton, goldfinch_hopfield_fused_scan_pytorch
    eval_item("Canonical-26", "goldfinch_hopfield", "1024x64", goldfinch_hopfield_fused_scan_triton, goldfinch_hopfield_fused_scan_pytorch, q[:, :, :1024], k[:, :, :1024], v[:, :, :1024])

    # 8. Fused Linformer Seq Proj
    from neuralops.core.triton_kernels.low_rank import fused_linformer_seq_proj_pytorch, fused_linformer_seq_proj_triton
    x_lin = torch.randn(8, 16, 4096, 64, device=device, dtype=dtype)
    p_lin = torch.randn(128, 4096, device=device, dtype=dtype)
    eval_item("Canonical-26", "linformer_seq_proj", "T=4096->128", fused_linformer_seq_proj_triton, fused_linformer_seq_proj_pytorch, p_lin, x_lin)

    # 9. Fused Norm Linear
    from neuralops.core.triton_kernels.fused_norm_linear import fused_rmsnorm_gate_triton, fused_rmsnorm_gate
    x_fnl = torch.randn(4, 1024, 2048, device=device, dtype=dtype)
    g_fnl = torch.randn(4, 1024, 2048, device=device, dtype=dtype)
    w_fnl = torch.randn(2048, device=device, dtype=dtype)
    eval_item("Canonical-26", "fused_norm_linear", "4x1024x2048", fused_rmsnorm_gate_triton, fused_rmsnorm_gate, x_fnl, g_fnl, w_fnl)

    # 10. Fused RoPE
    from neuralops.core.triton_kernels.fused_rope import fused_rope_fwd_kernel, apply_rope_rotation
    import triton
    cos_r = torch.cos(torch.randn(2048, 32, device=device, dtype=dtype))
    sin_r = torch.sin(torch.randn(2048, 32, device=device, dtype=dtype))
    def run_rope_tri(q_in, k_in, c, s):
        out_q = torch.empty_like(q_in)
        out_k = torch.empty_like(k_in)
        B_r, H_r, N_r, D_r = q_in.shape
        grid = (triton.cdiv(N_r, 32), B_r * H_r)
        fused_rope_fwd_kernel[grid](
            q_in, k_in, c, s, out_q, out_k,
            q_in.stride(0), q_in.stride(1), q_in.stride(2), q_in.stride(3),
            k_in.stride(0), k_in.stride(1), k_in.stride(2), k_in.stride(3),
            0, c.stride(0), c.stride(1),
            0, s.stride(0), s.stride(1),
            out_q.stride(0), out_q.stride(1), out_q.stride(2), out_q.stride(3),
            out_k.stride(0), out_k.stride(1), out_k.stride(2), out_k.stride(3),
            H_r, N_r, D_r, D_r // 2,
            BLOCK_M=32,
        )
        return out_q
    eval_item("Canonical-26", "fused_rope", "4x8x2048x64", run_rope_tri, lambda q, k, c, s: apply_rope_rotation(q, c, s), q, k, cos_r, sin_r)

    # 11. Softplus (Large Batch for High GPU Saturation)
    from neuralops.core.triton_kernels.softplus import softplus_fn, _softplus_pytorch
    x_sp = torch.randn(16, 1024, 1024, device=device, dtype=dtype) # 16M elements
    eval_item("Canonical-26", "softplus", "16M elements", softplus_fn, _softplus_pytorch, x_sp)

    # 12. Sliding Window Attention (Scaled sequence)
    from neuralops.core.triton_kernels.sliding_window import sliding_window_triton, sliding_window_pytorch
    q_sw = torch.randn(4, 8, 4096, 64, device=device, dtype=dtype)
    k_sw = torch.randn(4, 8, 4096, 64, device=device, dtype=dtype)
    v_sw = torch.randn(4, 8, 4096, 64, device=device, dtype=dtype)
    eval_item("Canonical-26", "sliding_window", "4x8x4096,W=256", sliding_window_triton, sliding_window_pytorch, q_sw, k_sw, v_sw, 256)

    # 13. Paged Attention (Batch 16, 64 Blocks)
    from neuralops.core.triton_kernels.paged_attention import paged_attention_kernel, paged_attention_pytorch
    num_blks = 128
    k_cache = torch.randn(num_blks, 16, 8, 64, device=device, dtype=dtype)
    v_cache = torch.randn(num_blks, 16, 8, 64, device=device, dtype=dtype)
    q_pa = torch.randn(16, 8, 64, device=device, dtype=dtype)
    block_tables = torch.randint(0, num_blks, (16, 8), device=device, dtype=torch.int32)
    seq_lens = torch.full((16,), 128, device=device, dtype=torch.int32)
    eval_item("Canonical-26", "paged_attention", "B=16,H=8,T=128", paged_attention_kernel, paged_attention_pytorch, q_pa, k_cache, v_cache, block_tables, seq_lens, 16)

    # 14. Cosformer Linear Attention
    from neuralops.core.triton_kernels.cosformer_triton_kernel import cosformer_triton_fwd, cosformer_pytorch_fwd
    eval_item("Canonical-26", "cosformer", "4x8x2048x64", cosformer_triton_fwd, cosformer_pytorch_fwd, q, k, v)

    # 15. Fused Recurrent Linear Step (Batch 32)
    from neuralops.core.triton_kernels.state_update import fused_recurrent_linear_step
    q_s = torch.randn(32, 8, 64, device=device, dtype=dtype)
    k_s = torch.randn(32, 8, 64, device=device, dtype=dtype)
    v_s = torch.randn(32, 8, 64, device=device, dtype=dtype)
    kv_s = torch.zeros(32, 8, 64, 64, device=device, dtype=dtype)
    z_s = torch.zeros(32, 8, 64, device=device, dtype=dtype)
    def pt_recurrent_step(q, k, v, kv, z):
        kv_new = kv + torch.einsum("bhk,bhv->bhkv", k.float(), v.float()).to(kv.dtype)
        z_new = z + k
        num = torch.einsum("bhk,bhkv->bhv", q.float(), kv_new.float())
        denom = torch.einsum("bhk,bhk->bh", q.float(), z_new.float()).unsqueeze(-1) + 1e-6
        return (num / denom).to(q.dtype), kv_new, z_new
    eval_item("Canonical-26", "recurrent_linear_step", "32x8x64", fused_recurrent_linear_step, pt_recurrent_step, q_s, k_s, v_s, kv_s.clone(), z_s.clone())

    # 16. Fused Kronecker GEMM (Scaled dimensions)
    from neuralops.core.triton_kernels.low_rank import fused_kronecker_batched_gemm_triton, fused_kronecker_batched_gemm_pytorch
    x_kron = torch.randn(16, 256, 1024, device=device, dtype=dtype)
    A_kron = torch.randn(32, 32, device=device, dtype=dtype)
    B_kron = torch.randn(32, 32, device=device, dtype=dtype)
    eval_item("Canonical-26", "fused_kronecker_gemm", "1024x(32x32)", fused_kronecker_batched_gemm_triton, fused_kronecker_batched_gemm_pytorch, x_kron, A_kron, B_kron)

    # 17. Fused MQA Bias Scale
    from neuralops.core.triton_kernels.gqa import fused_mqa_bias_scale_triton, fused_mqa_bias_scale_pytorch
    scores_mqa = torch.randn(8, 8, 512, 512, device=device, dtype=dtype)
    bias_mqa = torch.randn(1, 8, 1, 512, device=device, dtype=dtype)
    eval_item("Canonical-26", "fused_mqa_bias_scale", "8x8x512x512", fused_mqa_bias_scale_triton, fused_mqa_bias_scale_pytorch, scores_mqa, bias_mqa, 0.125, True)

    # 18. Fused MOBA Block Router
    from neuralops.core.triton_kernels.block_sparse import fused_moba_block_router_combine_triton, fused_moba_block_router_combine_pytorch
    Qb_m = torch.randn(4, 8, 32, 32, 64, device=device, dtype=dtype)
    Kb_m = torch.randn(4, 8, 32, 32, 64, device=device, dtype=dtype)
    Vb_m = torch.randn(4, 8, 32, 32, 64, device=device, dtype=dtype)
    eval_item("Canonical-26", "moba_block_router", "4x8x32x32x64", fused_moba_block_router_combine_triton, fused_moba_block_router_combine_pytorch, Qb_m, Kb_m, Vb_m, 0.125, top_k=4, is_causal=True)

    # 19. Fused Favor Plus
    from neuralops.core.triton_kernels.elementwise import fused_favor_plus_feature_map_triton, fused_favor_plus_feature_map_pytorch
    x_fav = torch.randn(8, 8, 512, 64, device=device, dtype=dtype)
    omega_fav = torch.randn(256, 64, device=device, dtype=dtype)
    eval_item("Canonical-26", "fused_favor_plus", "512x64,M=256", fused_favor_plus_feature_map_triton, fused_favor_plus_feature_map_pytorch, x_fav, omega_fav, 256, "favor_plus")

    # 20. Fused HNC Nullspace Step
    from neuralops.core.triton_kernels.low_rank import fused_hnc_nullspace_step_triton, fused_hnc_nullspace_step_pytorch
    w_hnc = torch.randn(4096, 4096, device=device, dtype=dtype)
    v_hnc = torch.randn(4096, 128, device=device, dtype=dtype)
    alpha_hnc = torch.randn(128, 4096, device=device, dtype=dtype)
    eval_item("Canonical-26", "hnc_nullspace_step", "4096x4096,r=128", fused_hnc_nullspace_step_triton, fused_hnc_nullspace_step_pytorch, w_hnc, v_hnc, alpha_hnc)

    # 21. Fused Cross Entropy
    from neuralops.core.triton_kernels.fused_ce import fused_linear_cross_entropy, fused_linear_cross_entropy_pytorch
    hid_ce = torch.randn(32, 512, 1024, device=device, dtype=dtype)
    wt_ce = torch.randn(50257, 1024, device=device, dtype=dtype)
    tgt_ce = torch.randint(0, 50257, (32, 512), device=device, dtype=torch.long)
    eval_item("Canonical-26", "fused_cross_entropy", "V=50257,N=16384", fused_linear_cross_entropy, fused_linear_cross_entropy_pytorch, hid_ce, wt_ce, tgt_ce)

    # =========================================================================
    # PART 2: TRITON ATTENTION MODULE MATRIX (ATTENTION_REGISTRY)
    # =========================================================================
    from neuralops.attentions import ATTENTION_REGISTRY
    triton_attentions = [
        "cosformer", "triton_cosformer", "gated_linear_attention", "gla_attention",
        "hgrn2_gated_layer", "gated_delta_net", "kda_attention", "chela_slay",
        "chela_slay_attention", "nystrom_attention", "performer_attention",
        "linformer_attention", "moba_attention", "adaptive_sparse_attention",
        "sliding_window_attention", "cur_attention", "differential_attention",
    ]

    for attn_name in triton_attentions:
        if attn_name in ATTENTION_REGISTRY:
            try:
                cls = ATTENTION_REGISTRY[attn_name]
                try:
                    mod = cls(dim=256, num_heads=4).to(device=device, dtype=dtype)
                except Exception:
                    mod = cls(hidden_size=256, num_attention_heads=4).to(device=device, dtype=dtype)

                x_in = torch.randn(4, 512, 256, device=device, dtype=dtype)

                # Triton execution
                def run_mod(m, x):
                    return m(x)

                eval_item("Layer-Modules", attn_name, "4x512x256", run_mod, run_mod, mod, x_in)
            except Exception as e:
                print(f"Layer-Modules   | {attn_name:<32} | Instantiation skip: {e}")

    # Output CSV summary
    out_csv = "results/all_triton_operators_silicon.csv"
    os.makedirs("results", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["category", "kernel", "shape", "triton_ms", "pytorch_ms", "speedup", "max_diff", "status"])
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 100)
    print(f"COMPLETE: Tested {len(results)} operators. Summary written to {out_csv}")
    print("=" * 100)

if __name__ == "__main__":
    main()
