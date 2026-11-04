#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
10-Phase CUDA Benchmark Suite
Executes end-to-end hardware validation across the 10 partitioned phases of CUDA kernels.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import torch

def benchmark_single(name, fn, *args, repeat=20, warmup=5):
    for _ in range(warmup):
        fn(*args)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeat):
            fn(*args)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / repeat
    else:
        t0 = time.perf_counter()
        for _ in range(repeat):
            fn(*args)
        return (time.perf_counter() - t0) * 1000.0 / repeat

def main():
    parser = argparse.ArgumentParser(description="10-Phase CUDA Benchmark")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="results/cuda_10_phases_benchmark.csv")
    args = parser.parse_args()

    print(f"Device: {args.device}, Output: {args.output}")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    results = []
    # Test key representative kernels across the 10 phases
    print("Benchmarking Phase 1 (Attention Decoding & Fast Serving)...")
    try:
        from neuralops.ops.cuda.fused_rope.rope_autograd import fused_rotary_emb
        x = torch.randn(2, 4, 1024, 64, device=args.device, dtype=torch.float16)
        cos = torch.randn(1, 1, 1024, 32, device=args.device, dtype=torch.float16)
        sin = torch.randn(1, 1, 1024, 32, device=args.device, dtype=torch.float16)
        ms = benchmark_single("fused_rope", fused_rotary_emb, x, cos, sin)
        results.append({"phase": 1, "module": "fused_rope", "latency_ms": ms, "status": "OK"})
        print(f"  fused_rope: {ms:.4f} ms")
    except Exception as e:
        results.append({"phase": 1, "module": "fused_rope", "latency_ms": 0.0, "status": str(e)})

    try:
        from neuralops.ops.cuda.flash_decoding.flash_dec_autograd import flash_decoding_attention
        q = torch.randn(2, 4, 1, 64, device=args.device, dtype=torch.float16)
        k = torch.randn(2, 4, 1024, 64, device=args.device, dtype=torch.float16)
        v = torch.randn(2, 4, 1024, 64, device=args.device, dtype=torch.float16)
        ms = benchmark_single("flash_decoding", flash_decoding_attention, q, k, v)
        results.append({"phase": 1, "module": "flash_decoding", "latency_ms": ms, "status": "OK"})
        print(f"  flash_decoding: {ms:.4f} ms")
    except Exception as e:
        results.append({"phase": 1, "module": "flash_decoding", "latency_ms": 0.0, "status": str(e)})

    print("Benchmarking Phase 2 (Context Parallel & Dynamic Convolutions)...")
    try:
        from neuralops.ops.cuda.sliding_window.sliding_autograd import sliding_window_attn
        q = torch.randn(2, 4, 1024, 64, device=args.device, dtype=torch.float16)
        k = torch.randn(2, 4, 1024, 64, device=args.device, dtype=torch.float16)
        v = torch.randn(2, 4, 1024, 64, device=args.device, dtype=torch.float16)
        ms = benchmark_single("sliding_window", sliding_window_attn, q, k, v, 128)
        results.append({"phase": 2, "module": "sliding_window", "latency_ms": ms, "status": "OK"})
        print(f"  sliding_window: {ms:.4f} ms")
    except Exception as e:
        results.append({"phase": 2, "module": "sliding_window", "latency_ms": 0.0, "status": str(e)})

    print("Benchmarking Phase 3 (State Space & Prefix Scans)...")
    try:
        from neuralops.ops.cuda.fms_scan.fms_autograd import fms_scan
        q = torch.randn(2, 4, 512, 64, device=args.device, dtype=torch.float16)
        k = torch.randn(2, 4, 512, 64, device=args.device, dtype=torch.float16)
        v = torch.randn(2, 4, 512, 64, device=args.device, dtype=torch.float16)
        g = torch.sigmoid(torch.randn(2, 4, 512, device=args.device, dtype=torch.float16))
        ms = benchmark_single("fms_scan", fms_scan, q, k, v, g)
        results.append({"phase": 3, "module": "fms_scan", "latency_ms": ms, "status": "OK"})
        print(f"  fms_scan: {ms:.4f} ms")
    except Exception as e:
        results.append({"phase": 3, "module": "fms_scan", "latency_ms": 0.0, "status": str(e)})

    print("Benchmarking Phase 7 (Continuous & Delta Scans)...")
    try:
        from neuralops.ops.cuda.gdn_scan.gdn_autograd import chunked_delta_gate_scan
        q = torch.randn(2, 4, 512, 64, device=args.device, dtype=torch.float16)
        k = torch.randn(2, 4, 512, 64, device=args.device, dtype=torch.float16)
        v = torch.randn(2, 4, 512, 64, device=args.device, dtype=torch.float16)
        b = torch.sigmoid(torch.randn(2, 4, 512, device=args.device, dtype=torch.float16))
        a = torch.sigmoid(torch.randn(2, 4, 512, device=args.device, dtype=torch.float16))
        ms = benchmark_single("gdn_scan", chunked_delta_gate_scan, q, k, v, b, a)
        results.append({"phase": 7, "module": "gdn_scan", "latency_ms": ms, "status": "OK"})
        print(f"  gdn_scan: {ms:.4f} ms")
    except Exception as e:
        results.append({"phase": 7, "module": "gdn_scan", "latency_ms": 0.0, "status": str(e)})

    print("Benchmarking Phase 9 (Recurrent Scans & GLA)...")
    try:
        from neuralops.ops.cuda.gla_scan.gla_autograd import gla_scan
        q = torch.randn(2, 4, 512, 64, device=args.device, dtype=torch.float16)
        k = torch.randn(2, 4, 512, 64, device=args.device, dtype=torch.float16)
        v = torch.randn(2, 4, 512, 64, device=args.device, dtype=torch.float16)
        g = torch.sigmoid(torch.randn(2, 4, 512, device=args.device, dtype=torch.float16))
        ms = benchmark_single("gla_scan", gla_scan, q, k, v, g)
        results.append({"phase": 9, "module": "gla_scan", "latency_ms": ms, "status": "OK"})
        print(f"  gla_scan: {ms:.4f} ms")
    except Exception as e:
        results.append({"phase": 9, "module": "gla_scan", "latency_ms": 0.0, "status": str(e)})

    try:
        from neuralops.ops.cuda.selective_scan.selective_scan_autograd import selective_scan
        u = torch.randn(2, 64, 512, device=args.device, dtype=torch.float16)
        delta = torch.randn(2, 64, 512, device=args.device, dtype=torch.float16).abs() + 0.1
        A = -torch.rand(64, 16, device=args.device, dtype=torch.float16) - 0.1
        B = torch.randn(2, 16, 512, device=args.device, dtype=torch.float16)
        C = torch.randn(2, 16, 512, device=args.device, dtype=torch.float16)
        D = torch.randn(64, device=args.device, dtype=torch.float16)
        ms = benchmark_single("selective_scan", selective_scan, u, delta, A, B, C, D)
        results.append({"phase": 9, "module": "selective_scan", "latency_ms": ms, "status": "OK"})
        print(f"  selective_scan: {ms:.4f} ms")
    except Exception as e:
        results.append({"phase": 9, "module": "selective_scan", "latency_ms": 0.0, "status": str(e)})

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["phase", "module", "latency_ms", "status"])
        writer.writeheader()
        writer.writerows(results)

    print(f"Results written to {args.output}")

if __name__ == "__main__":
    main()
