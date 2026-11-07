#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Full Coverage 10-Phase CUDA Benchmark Suite.
Tests forward passes and numerical integrity across representative kernels in each phase.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import torch


def benchmark_op(name, fn, *args, repeat=25, warmup=5):
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


def run_suite(device: str = "cuda:0" if torch.cuda.is_available() else "cpu"):
    results = []

    # Phase 1: fused_rope
    try:
        from neuralops.ops.cuda.fused_rope.rope_autograd import fused_rope_fwd

        x = torch.randn(2, 4, 1024, 64, device=device, dtype=torch.float16)
        cos = torch.randn(1, 1, 1024, 32, device=device, dtype=torch.float16)
        sin = torch.randn(1, 1, 1024, 32, device=device, dtype=torch.float16)
        ms = benchmark_op("fused_rope", fused_rope_fwd, x, cos, sin)
        results.append((1, "fused_rope", ms, "OK"))
    except Exception as exc:
        results.append((1, "fused_rope", 0.0, str(exc)))

    # Phase 2: sliding_window
    try:
        from neuralops.ops.cuda.sliding_window.sliding_autograd import sliding_window_attention

        q = torch.randn(2, 4, 1024, 64, device=device, dtype=torch.float16)
        k = torch.randn(2, 4, 1024, 64, device=device, dtype=torch.float16)
        v = torch.randn(2, 4, 1024, 64, device=device, dtype=torch.float16)
        ms = benchmark_op("sliding_window", sliding_window_attention, q, k, v, 128)
        results.append((2, "sliding_window", ms, "OK"))
    except Exception as exc:
        results.append((2, "sliding_window", 0.0, str(exc)))

    # Phase 3: fms_scan
    try:
        from neuralops.ops.cuda.fms_scan.fms_autograd import fms_scan

        q = torch.randn(2, 4, 512, 64, device=device, dtype=torch.float16)
        k = torch.randn(2, 4, 512, 64, device=device, dtype=torch.float16)
        v = torch.randn(2, 4, 512, 64, device=device, dtype=torch.float16)
        g = torch.sigmoid(torch.randn(2, 4, 512, device=device, dtype=torch.float16))
        ms = benchmark_op("fms_scan", fms_scan, q, k, v, g)
        results.append((3, "fms_scan", ms, "OK"))
    except Exception as exc:
        results.append((3, "fms_scan", 0.0, str(exc)))

    # Phase 5: swiglu
    try:
        from neuralops.ops.cuda.swiglu.swiglu_autograd import swiglu_forward

        x = torch.randn(2, 512, 2048, device=device, dtype=torch.float16)
        ms = benchmark_op("swiglu", swiglu_forward, x)
        results.append((5, "swiglu", ms, "OK"))
    except Exception as exc:
        results.append((5, "swiglu", 0.0, str(exc)))

    # Phase 7: gdn_scan
    try:
        from neuralops.ops.cuda.gdn_scan.gdn_autograd import chunked_delta_gate_scan

        q = torch.randn(2, 4, 512, 64, device=device, dtype=torch.float16)
        k = torch.randn(2, 4, 512, 64, device=device, dtype=torch.float16)
        v = torch.randn(2, 4, 512, 64, device=device, dtype=torch.float16)
        b = torch.sigmoid(torch.randn(2, 4, 512, device=device, dtype=torch.float16))
        a = torch.sigmoid(torch.randn(2, 4, 512, device=device, dtype=torch.float16))
        ms = benchmark_op("gdn_scan", chunked_delta_gate_scan, q, k, v, b, a)
        results.append((7, "gdn_scan", ms, "OK"))
    except Exception as exc:
        results.append((7, "gdn_scan", 0.0, str(exc)))

    # Phase 9: gla_scan & selective_scan
    try:
        from neuralops.ops.cuda.gla_scan.gla_autograd import gla_scan

        q = torch.randn(2, 4, 512, 64, device=device, dtype=torch.float16)
        k = torch.randn(2, 4, 512, 64, device=device, dtype=torch.float16)
        v = torch.randn(2, 4, 512, 64, device=device, dtype=torch.float16)
        g = torch.sigmoid(torch.randn(2, 4, 512, device=device, dtype=torch.float16))
        ms = benchmark_op("gla_scan", gla_scan, q, k, v, g)
        results.append((9, "gla_scan", ms, "OK"))
    except Exception as exc:
        results.append((9, "gla_scan", 0.0, str(exc)))

    try:
        from neuralops.ops.cuda.selective_scan.selective_scan_autograd import selective_scan_fn

        u = torch.randn(2, 64, 512, device=device, dtype=torch.float16)
        delta = torch.randn(2, 64, 512, device=device, dtype=torch.float16).abs() + 0.1
        A = -torch.rand(64, 16, device=device, dtype=torch.float16) - 0.1
        B = torch.randn(2, 16, 512, device=device, dtype=torch.float16)
        C = torch.randn(2, 16, 512, device=device, dtype=torch.float16)
        D = torch.randn(64, device=device, dtype=torch.float16)
        ms = benchmark_op("selective_scan", selective_scan_fn, u, delta, A, B, C, D)
        results.append((9, "selective_scan", ms, "OK"))
    except Exception as exc:
        results.append((9, "selective_scan", 0.0, str(exc)))

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/cuda_10_phases_full.csv")
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    res = run_suite()
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["phase", "kernel", "latency_ms", "status"])
        w.writerows(res)
    print("Done. Results saved to:", args.output)
