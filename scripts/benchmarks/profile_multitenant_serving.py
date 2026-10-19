#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Profile multitenant serving module.
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
    print("=== MULTI-TENANT 4-STREAM INFERENCE SERVING & CONTEXT SWITCHING PROFILE ===")
    device = torch.device("cuda:0")
    device_name = torch.cuda.get_device_name(0)
    print(f"Device: {device_name}")

    tenant_streams = [torch.cuda.Stream(device=device) for _ in range(4)]

    print("Allocating resident model weights (2.0 GB)...")
    base_weights = [torch.empty(256 * 1024 * 512, dtype=torch.float32, device=device) for _ in range(4)]

    B1, H1, S1, D1 = 2, 4, 1024, 64
    q1 = torch.randn(B1, H1, S1, D1, device=device, dtype=torch.float16)
    k1 = torch.randn(B1, H1, S1, D1, device=device, dtype=torch.float16)
    v1 = torch.randn(B1, H1, S1, D1, device=device, dtype=torch.float16)

    B2, D2, L2, N2 = 2, 64, 1024, 16
    u2 = torch.randn(B2, D2, L2, device=device, dtype=torch.float32)
    delta2 = F.softplus(torch.randn(B2, D2, L2, device=device, dtype=torch.float32) - 4.0)
    A2 = -torch.exp(torch.randn(D2, N2, device=device, dtype=torch.float32))
    B2_p = torch.randn(B2, N2, L2, device=device, dtype=torch.float32) * 0.1
    C2_p = torch.randn(B2, N2, L2, device=device, dtype=torch.float32) * 0.1

    M3, K3, R3 = 2048, 768, 64
    X3 = torch.randn(M3, K3, device=device, dtype=torch.float16)
    V_r3 = torch.randn(R3, K3, device=device, dtype=torch.float16)
    U_r3 = torch.randn(K3, R3, device=device, dtype=torch.float16)
    alpha3 = torch.randn(R3, device=device, dtype=torch.float16)

    A4 = torch.randn(1024, 1024, device=device, dtype=torch.float16)
    B4 = torch.randn(1024, 1024, device=device, dtype=torch.float16)

    print("Warming up 4 tenant streams...")
    for _ in range(25):
        with torch.cuda.stream(tenant_streams[0]):
            w1 = flash_gqa(q1, k1, v1, is_causal=True)
        with torch.cuda.stream(tenant_streams[1]):
            w2, _ = selective_scan(u2, delta2, A2, B2_p, C2_p)
        with torch.cuda.stream(tenant_streams[2]):
            w3 = fused_lowrank_gated_gemm(X3, V_r3, U_r3, alpha3)
        with torch.cuda.stream(tenant_streams[3]):
            w4 = torch.matmul(A4, B4)
        torch.cuda.synchronize()
        del w1, w2, w3, w4

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    _mem_initial = get_cuda_mem()

    NUM_ROUNDS = 200
    print(f"\nBenchmarking Sequential vs Multi-Tenant Serving ({NUM_ROUNDS} rounds)...")

    torch.cuda.synchronize()
    t0_seq = time.perf_counter()
    for _ in range(NUM_ROUNDS):
        ref1 = flash_gqa(q1, k1, v1, is_causal=True)
        ref2, _ = selective_scan(u2, delta2, A2, B2_p, C2_p)
        ref3 = fused_lowrank_gated_gemm(X3, V_r3, U_r3, alpha3)
        ref4 = torch.matmul(A4, B4)
        del ref1, ref2, ref3, ref4
    torch.cuda.synchronize()
    t_seq_ms = (time.perf_counter() - t0_seq) / NUM_ROUNDS * 1000

    torch.cuda.synchronize()
    t0_par = time.perf_counter()
    checkpoint_frags = []

    for round_idx in range(1, NUM_ROUNDS + 1):
        with torch.cuda.stream(tenant_streams[0]):
            p1 = flash_gqa(q1, k1, v1, is_causal=True)
        with torch.cuda.stream(tenant_streams[1]):
            p2, _ = selective_scan(u2, delta2, A2, B2_p, C2_p)
        with torch.cuda.stream(tenant_streams[2]):
            p3 = fused_lowrank_gated_gemm(X3, V_r3, U_r3, alpha3)
        with torch.cuda.stream(tenant_streams[3]):
            p4 = torch.matmul(A4, B4)

        torch.cuda.synchronize()

        if round_idx in [1, 50, 100, 150, 200]:
            alloc_mb = torch.cuda.memory_allocated() / (1024**2)
            res_mb = torch.cuda.memory_reserved() / (1024**2)
            frag_ratio = 1.0 - (alloc_mb / res_mb)
            checkpoint_frags.append(frag_ratio)

        del p1, p2, p3, p4

    torch.cuda.synchronize()
    t_par_ms = (time.perf_counter() - t0_par) / NUM_ROUNDS * 1000

    mem_final = get_cuda_mem()
    alloc_mb = mem_final["allocated_mb"]
    res_mb = mem_final["reserved_mb"]
    peak_alloc_mb = mem_final["max_allocated_mb"]
    peak_res_mb = mem_final["max_reserved_mb"]

    frag_ratio_final = 1.0 - (alloc_mb / (res_mb + 1e-6))
    frag_pct = frag_ratio_final * 100.0

    serving_speedup = t_seq_ms / t_par_ms

    print("\n--- MULTI-TENANT SERVING SUMMARY ---")
    print(f"Sequential Latency:      {t_seq_ms:.4f} ms/query batch")
    print(f"Concurrent Latency:      {t_par_ms:.4f} ms/query batch")
    print(f"Serving Overlap Speedup: {serving_speedup:.3f}x")
    print(f"Base Allocated VRAM:     {alloc_mb:.2f} MB")
    print(f"Peak Allocated VRAM:     {peak_alloc_mb:.2f} MB")
    print(f"Peak Reserved Pool:      {peak_res_mb:.2f} MB")
    print(f"Allocator Fragmentation: {frag_pct:.2f}% (< 3.0%: {frag_pct < 3.0})")

    del base_weights
    gc.collect()
    torch.cuda.empty_cache()

    results = {
        "device": device_name,
        "tenants": 4,
        "num_rounds": NUM_ROUNDS,
        "workload": {
            "tenant_1": "FlashAttention (FP16)",
            "tenant_2": "Selective SSM Scan (FP32)",
            "tenant_3": "Low-Rank SVD GEMM (FP16)",
            "tenant_4": "Matmul Decomposition (FP16)",
        },
        "performance": {
            "sequential_latency_ms": round(t_seq_ms, 4),
            "concurrent_latency_ms": round(t_par_ms, 4),
            "serving_speedup": round(serving_speedup, 3),
        },
        "vram_allocator": {
            "allocated_mb": alloc_mb,
            "peak_allocated_mb": peak_alloc_mb,
            "reserved_mb": res_mb,
            "peak_reserved_mb": peak_res_mb,
            "fragmentation_ratio": round(frag_ratio_final, 4),
            "fragmentation_pct": round(frag_pct, 2),
            "strictly_below_3_pct": bool(frag_pct < 3.0),
        },
    }

    out_file = "/mnt/d/SVD_linear_Attention/results/profiling/multitenant_serving_profile.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved profile results to {out_file}")


if __name__ == "__main__":
    main()
