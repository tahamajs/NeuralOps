#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Stress 1000 multistream module.
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
    print("=== STARTING 1,000-ITERATION VRAM CONCURRENCY & LONG-DURATION STRESS TEST ===")
    device = torch.device("cuda:0")
    device_name = torch.cuda.get_device_name(0)
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"Target GPU: {device_name} ({total_mem_gb:.2f} GB VRAM)")

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

    print("Allocating heavy background memory load (4.5 GB)...")
    background_tensors = []
    num_chunks = 9
    for _ in range(num_chunks):
        background_tensors.append(torch.empty(256 * 1024 * 512, dtype=torch.float32, device=device))

    torch.cuda.synchronize()
    bg_mem = get_cuda_mem()
    print(
        f"Background load active: Allocated = {bg_mem['allocated_mb']:.2f} MB, Reserved = {bg_mem['reserved_mb']:.2f} MB"
    )

    print("Warming up concurrent kernels under load...")
    for _ in range(25):
        with torch.cuda.stream(stream1):
            w1 = flash_gqa(q, k, v, is_causal=True)
        with torch.cuda.stream(stream2):
            w2, _ = selective_scan(u, delta, A, B_param, C_param)
        with torch.cuda.stream(stream3):
            w3 = fused_lowrank_gated_gemm(X, V_r, U_r, alpha)
        torch.cuda.synchronize()
        del w1, w2, w3

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline_mem = get_cuda_mem()

    NUM_ITERS = 1000
    print("\nLaunching 1,000-iteration multi-stream stress test under heavy memory pressure...")

    _checkpoints = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    checkpoint_stats = {}

    total_nans = 0
    total_infs = 0

    t0 = time.time()
    for it in range(1, NUM_ITERS + 1):
        with torch.cuda.stream(stream1):
            p1 = flash_gqa(q, k, v, is_causal=True)
        with torch.cuda.stream(stream2):
            p2, _ = selective_scan(u, delta, A, B_param, C_param)
        with torch.cuda.stream(stream3):
            p3 = fused_lowrank_gated_gemm(X, V_r, U_r, alpha)

        torch.cuda.synchronize()

        if it % 100 == 0:
            nan_cnt = (
                int(torch.isnan(p1).sum().item())
                + int(torch.isnan(p2).sum().item())
                + int(torch.isnan(p3).sum().item())
            )
            inf_cnt = (
                int(torch.isinf(p1).sum().item())
                + int(torch.isinf(p2).sum().item())
                + int(torch.isinf(p3).sum().item())
            )
            total_nans += nan_cnt
            total_infs += inf_cnt

            mem = get_cuda_mem()
            checkpoint_stats[it] = mem
            drift_from_100 = (
                round(mem["allocated_mb"] - checkpoint_stats[100]["allocated_mb"], 4)
                if 100 in checkpoint_stats
                else 0.0
            )
            print(
                f"  Iter {it:4d}/1000: Allocated = {mem['allocated_mb']:.2f} MB, Reserved = {mem['reserved_mb']:.2f} MB | Drift from iter 100: {drift_from_100:.4f} MB"
            )

        del p1, p2, p3

    torch.cuda.synchronize()
    elapsed = time.time() - t0
    final_mem = get_cuda_mem()

    net_drift = round(checkpoint_stats[1000]["allocated_mb"] - checkpoint_stats[100]["allocated_mb"], 4)
    peak_allocated = final_mem["max_allocated_mb"]
    peak_reserved = final_mem["max_reserved_mb"]

    frag_ratio = 1.0 - (final_mem["allocated_mb"] / (final_mem["reserved_mb"] + 1e-6))

    print("\n--- 1,000-ITERATION STRESS SUMMARY ---")
    print(f"Total Elapsed Time:    {elapsed:.2f} s ({elapsed / NUM_ITERS * 1000:.3f} ms/iter)")
    print(f"Baseline Allocated:    {baseline_mem['allocated_mb']:.2f} MB")
    print(f"Final Allocated:       {final_mem['allocated_mb']:.2f} MB")
    print(f"Peak Allocated:        {peak_allocated:.2f} MB")
    print(f"Peak Reserved:         {peak_reserved:.2f} MB")
    print(f"Net Memory Drift:      {net_drift:.4f} MB (Strict Zero Leak: {net_drift == 0.0})")
    print(f"CUDA Fragmentation:    {frag_ratio:.4f}")
    print(f"Total NaNs Detected:   {total_nans}")
    print(f"Total Infs Detected:   {total_infs}")

    del background_tensors
    gc.collect()
    torch.cuda.empty_cache()

    stress_report = {
        "device": device_name,
        "total_vram_gb": round(total_mem_gb, 2),
        "iterations": NUM_ITERS,
        "background_load_mb": bg_mem["allocated_mb"],
        "elapsed_sec": round(elapsed, 2),
        "ms_per_iter": round((elapsed / NUM_ITERS) * 1000, 3),
        "baseline_allocated_mb": baseline_mem["allocated_mb"],
        "final_allocated_mb": final_mem["allocated_mb"],
        "peak_allocated_mb": peak_allocated,
        "peak_reserved_mb": peak_reserved,
        "memory_drift_iter100_to_1000_mb": net_drift,
        "zero_leak_verified": (net_drift == 0.0),
        "cuda_allocator_fragmentation_ratio": round(frag_ratio, 4),
        "numerical_integrity": {
            "total_nans": total_nans,
            "total_infs": total_infs,
            "clean": (total_nans == 0 and total_infs == 0),
        },
        "checkpoints": checkpoint_stats,
    }

    out_file = "/mnt/d/SVD_linear_Attention/results/profiling/vram_concurrency_1000_stress.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(stress_report, f, indent=2)
    print(f"Saved stress profile to {out_file}")


if __name__ == "__main__":
    main()
