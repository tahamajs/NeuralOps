# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Multi stream benchmark module.
"""

import os
import sys
import gc
import json
import time
import torch
import torch.nn.functional as F

sys.path.insert(0, "/mnt/d/SVD_linear_Attention")
from neuralops.core.cuda_kernels.flash_gqa.flash_gqa_autograd import flash_gqa
from neuralops.core.cuda_kernels.selective_scan import selective_scan
from neuralops.core.cuda_kernels.low_rank_svd import fused_lowrank_gated_gemm


def get_cuda_mem():
    """Execute get cuda mem routine."""
    return {
        "allocated_mb": round(torch.cuda.memory_allocated() / (1024**2), 4),
        "max_allocated_mb": round(torch.cuda.max_memory_allocated() / (1024**2), 4),
        "reserved_mb": round(torch.cuda.memory_reserved() / (1024**2), 4),
        "max_reserved_mb": round(torch.cuda.max_memory_reserved() / (1024**2), 4),
    }


def main():
    """Execute main routine."""
    print("=== MULTI-STREAM GPU CONCURRENCY & OVERLAP PROFILING ===")
    device = torch.device("cuda:0")
    device_name = torch.cuda.get_device_name(0)
    print(f"Device: {device_name}")

    stream1 = torch.cuda.Stream(device=device)
    stream2 = torch.cuda.Stream(device=device)
    stream3 = torch.cuda.Stream(device=device)

    B1, H1, S1, D1 = 4, 8, 1024, 64
    q = torch.randn(B1, H1, S1, D1, device=device, dtype=torch.float16)
    k = torch.randn(B1, H1, S1, D1, device=device, dtype=torch.float16)
    v = torch.randn(B1, H1, S1, D1, device=device, dtype=torch.float16)

    B2, D2, L2, N2 = 4, 64, 1024, 16
    u = torch.randn(B2, D2, L2, device=device, dtype=torch.float32)
    delta = F.softplus(torch.randn(B2, D2, L2, device=device, dtype=torch.float32) - 4.0)
    A = -torch.exp(torch.randn(D2, N2, device=device, dtype=torch.float32))
    B_param = torch.randn(B2, N2, L2, device=device, dtype=torch.float32) * 0.1
    C_param = torch.randn(B2, N2, L2, device=device, dtype=torch.float32) * 0.1

    M3, K3, R3 = 4096, 768, 64
    X = torch.randn(M3, K3, device=device, dtype=torch.float16)
    V_r = torch.randn(R3, K3, device=device, dtype=torch.float16)
    U_r = torch.randn(K3, R3, device=device, dtype=torch.float16)
    alpha = torch.randn(R3, device=device, dtype=torch.float16)

    print("Warming up kernels...")
    for _ in range(25):
        with torch.cuda.stream(stream1):
            w1 = flash_gqa(q, k, v, is_causal=True)
        with torch.cuda.stream(stream2):
            w2, _ = selective_scan(u, delta, A, B_param, C_param)
        with torch.cuda.stream(stream3):
            w3 = fused_lowrank_gated_gemm(X, V_r, U_r, alpha)
        torch.cuda.synchronize()
        del w1, w2, w3

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    NUM_ITERS = 100
    print("\nRunning 100-iteration standalone measurements...")

    ev_start1 = torch.cuda.Event(enable_timing=True)
    ev_end1 = torch.cuda.Event(enable_timing=True)
    ev_start1.record(stream1)
    for _ in range(NUM_ITERS):
        with torch.cuda.stream(stream1):
            tmp1 = flash_gqa(q, k, v, is_causal=True)
        del tmp1
    ev_end1.record(stream1)
    stream1.synchronize()
    t_stream1_ms = ev_start1.elapsed_time(ev_end1) / NUM_ITERS

    ev_start2 = torch.cuda.Event(enable_timing=True)
    ev_end2 = torch.cuda.Event(enable_timing=True)
    ev_start2.record(stream2)
    for _ in range(NUM_ITERS):
        with torch.cuda.stream(stream2):
            tmp2, _ = selective_scan(u, delta, A, B_param, C_param)
        del tmp2
    ev_end2.record(stream2)
    stream2.synchronize()
    t_stream2_ms = ev_start2.elapsed_time(ev_end2) / NUM_ITERS

    ev_start3 = torch.cuda.Event(enable_timing=True)
    ev_end3 = torch.cuda.Event(enable_timing=True)
    ev_start3.record(stream3)
    for _ in range(NUM_ITERS):
        with torch.cuda.stream(stream3):
            tmp3 = fused_lowrank_gated_gemm(X, V_r, U_r, alpha)
        del tmp3
    ev_end3.record(stream3)
    stream3.synchronize()
    t_stream3_ms = ev_start3.elapsed_time(ev_end3) / NUM_ITERS

    sum_individual_ms = t_stream1_ms + t_stream2_ms + t_stream3_ms
    max_individual_ms = max(t_stream1_ms, t_stream2_ms, t_stream3_ms)

    print("Standalone Latencies (avg per iteration):")
    print(f"  Stream 1 (FlashAttention): {t_stream1_ms:.4f} ms")
    print(f"  Stream 2 (Selective Scan): {t_stream2_ms:.4f} ms")
    print(f"  Stream 3 (WMMA Low-Rank):  {t_stream3_ms:.4f} ms")
    print(f"  Sum of individual times:   {sum_individual_ms:.4f} ms")
    print(f"  Critical path (max single): {max_individual_ms:.4f} ms")

    torch.cuda.synchronize()
    t0_seq = time.perf_counter()
    ev_seq_start = torch.cuda.Event(enable_timing=True)
    ev_seq_end = torch.cuda.Event(enable_timing=True)
    ev_seq_start.record()
    for _ in range(NUM_ITERS):
        ref1 = flash_gqa(q, k, v, is_causal=True)
        ref2, _ = selective_scan(u, delta, A, B_param, C_param)
        ref3 = fused_lowrank_gated_gemm(X, V_r, U_r, alpha)
        del ref1, ref2, ref3
    ev_seq_end.record()
    torch.cuda.synchronize()
    t_seq_total_wall_sec = time.perf_counter() - t0_seq
    t_seq_cuda_ms = ev_seq_start.elapsed_time(ev_seq_end) / NUM_ITERS

    ref1 = flash_gqa(q, k, v, is_causal=True)
    ref2, _ = selective_scan(u, delta, A, B_param, C_param)
    ref3 = fused_lowrank_gated_gemm(X, V_r, U_r, alpha)

    torch.cuda.synchronize()
    mem_initial = get_cuda_mem()

    ev_par_start = torch.cuda.Event(enable_timing=True)
    ev_par_end = torch.cuda.Event(enable_timing=True)

    t0_par = time.perf_counter()
    ev_par_start.record(stream1)

    checkpoint_mems = {}
    for i in range(1, NUM_ITERS + 1):
        with torch.cuda.stream(stream1):
            par1 = flash_gqa(q, k, v, is_causal=True)
        with torch.cuda.stream(stream2):
            par2, _ = selective_scan(u, delta, A, B_param, C_param)
        with torch.cuda.stream(stream3):
            par3 = fused_lowrank_gated_gemm(X, V_r, U_r, alpha)

        torch.cuda.synchronize()
        if i in [1, 25, 50, 75, 100]:
            checkpoint_mems[i] = get_cuda_mem()

        if i < NUM_ITERS:
            del par1, par2, par3

    ev_s2 = torch.cuda.Event()
    ev_s3 = torch.cuda.Event()
    ev_s2.record(stream2)
    ev_s3.record(stream3)
    stream1.wait_event(ev_s2)
    stream1.wait_event(ev_s3)
    ev_par_end.record(stream1)

    torch.cuda.synchronize()
    t_par_total_wall_sec = time.perf_counter() - t0_par
    t_par_cuda_ms = ev_par_start.elapsed_time(ev_par_end) / NUM_ITERS

    nan1 = int(torch.isnan(par1).sum().item())
    inf1 = int(torch.isinf(par1).sum().item())
    diff1 = float(torch.max(torch.abs(ref1 - par1)).item())

    nan2 = int(torch.isnan(par2).sum().item())
    inf2 = int(torch.isinf(par2).sum().item())
    diff2 = float(torch.max(torch.abs(ref2 - par2)).item())

    nan3 = int(torch.isnan(par3).sum().item())
    inf3 = int(torch.isinf(par3).sum().item())
    diff3 = float(torch.max(torch.abs(ref3 - par3)).item())

    del ref1, ref2, ref3, par1, par2, par3
    torch.cuda.synchronize()
    mem_final = get_cuda_mem()

    mem_drift_1_to_100 = round(checkpoint_mems[100]["allocated_mb"] - checkpoint_mems[1]["allocated_mb"], 4)

    speedup_cuda = t_seq_cuda_ms / max(t_par_cuda_ms, 1e-6)
    speedup_wall = t_seq_total_wall_sec / max(t_par_total_wall_sec, 1e-6)

    overlap_efficiency = (sum_individual_ms - t_par_cuda_ms) / max(sum_individual_ms - max_individual_ms, 1e-6)
    overlap_pct = max(0.0, min(100.0, overlap_efficiency * 100.0))

    results = {
        "device": device_name,
        "iterations": NUM_ITERS,
        "workload": {
            "stream1_flash_attn": {"batch": B1, "heads": H1, "seq_len": S1, "head_dim": D1, "dtype": "fp16"},
            "stream2_selective_scan": {"batch": B2, "dim": D2, "seq_len": L2, "d_state": N2, "dtype": "fp32"},
            "stream3_lowrank_svd": {"M": M3, "K": K3, "R": R3, "dtype": "fp16"},
        },
        "standalone_latencies_ms": {
            "stream1_flash_attn": round(t_stream1_ms, 4),
            "stream2_selective_scan": round(t_stream2_ms, 4),
            "stream3_lowrank_svd": round(t_stream3_ms, 4),
            "sum_individual_ms": round(sum_individual_ms, 4),
            "critical_path_ms": round(max_individual_ms, 4),
        },
        "benchmark_timings": {
            "sequential_cuda_ms_per_iter": round(t_seq_cuda_ms, 4),
            "concurrent_cuda_ms_per_iter": round(t_par_cuda_ms, 4),
            "speedup_cuda_events": round(speedup_cuda, 3),
            "sequential_wall_total_sec": round(t_seq_total_wall_sec, 4),
            "concurrent_wall_total_sec": round(t_par_total_wall_sec, 4),
            "speedup_wall_clock": round(speedup_wall, 3),
            "stream_overlap_efficiency_pct": round(overlap_pct, 2),
        },
        "numerical_integrity": {
            "stream1": {"nans": nan1, "infs": inf1, "max_diff_vs_seq": diff1},
            "stream2": {"nans": nan2, "infs": inf2, "max_diff_vs_seq": diff2},
            "stream3": {"nans": nan3, "infs": inf3, "max_diff_vs_seq": diff3},
            "clean_numerical_integrity": (
                nan1 == 0
                and inf1 == 0
                and diff1 == 0.0
                and nan2 == 0
                and inf2 == 0
                and diff2 == 0.0
                and nan3 == 0
                and inf3 == 0
                and diff3 == 0.0
            ),
        },
        "vram_memory": {
            "initial_mb": mem_initial["allocated_mb"],
            "final_mb": mem_final["allocated_mb"],
            "peak_allocated_mb": mem_final["max_allocated_mb"],
            "peak_reserved_mb": mem_final["max_reserved_mb"],
            "drift_iter1_to_100_mb": mem_drift_1_to_100,
            "checkpoints": checkpoint_mems,
            "zero_leak_verified": (mem_drift_1_to_100 == 0.0),
        },
    }

    print("\n--- RESULTS SUMMARY ---")
    print(f"Sequential CUDA Time:   {t_seq_cuda_ms:.4f} ms/iter")
    print(f"Concurrent CUDA Time:   {t_par_cuda_ms:.4f} ms/iter")
    print(f"CUDA Event Speedup:     {speedup_cuda:.3f}x")
    print(f"Wall Clock Speedup:     {speedup_wall:.3f}x")
    print(f"Stream Overlap Eff:     {overlap_pct:.2f}%")
    print(f"Total NaNs: {nan1 + nan2 + nan3} | Total Infs: {inf1 + inf2 + inf3}")
    print(f"Bitwise Match: diff1={diff1}, diff2={diff2}, diff3={diff3}")
    print(f"Memory Drift (Iter 1 -> 100): {mem_drift_1_to_100} MB (Zero Leak: {mem_drift_1_to_100 == 0.0})")

    out_json = "/mnt/d/SVD_linear_Attention/results/profiling/multi_stream_concurrency_profile.json"
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {out_json}")


if __name__ == "__main__":
    main()
