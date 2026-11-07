#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Hardware Benchmark Measuring Native CUDA Speedup vs PyTorch Reference on GPU.
"""

from __future__ import annotations

import argparse
import csv
import os

import torch
import torch.nn.functional as F


def time_fn(fn, *args, repeat=20, warmup=5):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/cuda_10_phases_speedup_rtx3090.csv")
    args = parser.parse_args()

    device = "cuda:0"
    results = []

    print("Measuring CUDA vs PyTorch across 10 Phases...")

    # Phase 1: fused_rope
    try:
        from neuralops.ops.cuda.fused_rope.rope_autograd import fused_rotary_emb, fused_rotary_emb_pytorch_ref

        x = torch.randn(2, 4, 1024, 64, device=device, dtype=torch.float16)
        cos = torch.randn(1, 1, 1024, 32, device=device, dtype=torch.float16)
        sin = torch.randn(1, 1, 1024, 32, device=device, dtype=torch.float16)
        cuda_ms = time_fn(fused_rotary_emb, x, cos, sin)
        py_ms = time_fn(fused_rotary_emb_pytorch_ref, x, cos, sin)
        speedup = py_ms / (cuda_ms + 1e-9)
        results.append(
            {
                "phase": 1,
                "module": "fused_rope",
                "cuda_ms": cuda_ms,
                "pytorch_ms": py_ms,
                "speedup": speedup,
                "status": "OK",
            }
        )
        print(f"  [Phase 1] fused_rope: CUDA={cuda_ms:.4f}ms, PyTorch={py_ms:.4f}ms, Speedup={speedup:.2f}x")
    except Exception as e:
        results.append(
            {"phase": 1, "module": "fused_rope", "cuda_ms": 0.0, "pytorch_ms": 0.0, "speedup": 0.0, "status": str(e)}
        )

    # Phase 2: sliding_window
    try:
        from neuralops.ops.cuda.sliding_window.sliding_autograd import sliding_window_attn, sliding_window_pytorch_ref

        q = torch.randn(2, 4, 1024, 64, device=device, dtype=torch.float16)
        k = torch.randn(2, 4, 1024, 64, device=device, dtype=torch.float16)
        v = torch.randn(2, 4, 1024, 64, device=device, dtype=torch.float16)
        cuda_ms = time_fn(sliding_window_attn, q, k, v, 128)
        py_ms = time_fn(sliding_window_pytorch_ref, q, k, v, 128)
        speedup = py_ms / (cuda_ms + 1e-9)
        results.append(
            {
                "phase": 2,
                "module": "sliding_window",
                "cuda_ms": cuda_ms,
                "pytorch_ms": py_ms,
                "speedup": speedup,
                "status": "OK",
            }
        )
        print(f"  [Phase 2] sliding_window: CUDA={cuda_ms:.4f}ms, PyTorch={py_ms:.4f}ms, Speedup={speedup:.2f}x")
    except Exception as e:
        results.append(
            {
                "phase": 2,
                "module": "sliding_window",
                "cuda_ms": 0.0,
                "pytorch_ms": 0.0,
                "speedup": 0.0,
                "status": str(e),
            }
        )

    # Phase 3: fms_scan
    try:
        from neuralops.ops.cuda.fms_scan.fms_autograd import fms_scan, fms_scan_pytorch_ref

        q = torch.randn(2, 4, 512, 64, device=device, dtype=torch.float16)
        k = torch.randn(2, 4, 512, 64, device=device, dtype=torch.float16)
        v = torch.randn(2, 4, 512, 64, device=device, dtype=torch.float16)
        g = torch.sigmoid(torch.randn(4, device=device, dtype=torch.float16))
        cuda_ms = time_fn(fms_scan, q, k, v, g)
        py_ms = time_fn(fms_scan_pytorch_ref, q, k, v, g)
        speedup = py_ms / (cuda_ms + 1e-9)
        results.append(
            {
                "phase": 3,
                "module": "fms_scan",
                "cuda_ms": cuda_ms,
                "pytorch_ms": py_ms,
                "speedup": speedup,
                "status": "OK",
            }
        )
        print(f"  [Phase 3] fms_scan: CUDA={cuda_ms:.4f}ms, PyTorch={py_ms:.4f}ms, Speedup={speedup:.2f}x")
    except Exception as e:
        results.append(
            {"phase": 3, "module": "fms_scan", "cuda_ms": 0.0, "pytorch_ms": 0.0, "speedup": 0.0, "status": str(e)}
        )

    # Phase 7: gdn_scan
    try:
        from neuralops.ops.cuda.gdn_scan.gdn_autograd import (
            chunked_delta_gate_scan,
            chunked_delta_gate_scan_pytorch_ref,
        )

        q = torch.randn(2, 4, 512, 64, device=device, dtype=torch.float16)
        k = torch.randn(2, 4, 512, 64, device=device, dtype=torch.float16)
        v = torch.randn(2, 4, 512, 64, device=device, dtype=torch.float16)
        b = torch.sigmoid(torch.randn(2, 4, 512, device=device, dtype=torch.float16))
        a = torch.sigmoid(torch.randn(2, 4, 512, device=device, dtype=torch.float16))
        cuda_ms = time_fn(chunked_delta_gate_scan, q, k, v, b, a)
        py_ms = time_fn(chunked_delta_gate_scan_pytorch_ref, q, k, v, b, a)
        speedup = py_ms / (cuda_ms + 1e-9)
        results.append(
            {
                "phase": 7,
                "module": "gdn_scan",
                "cuda_ms": cuda_ms,
                "pytorch_ms": py_ms,
                "speedup": speedup,
                "status": "OK",
            }
        )
        print(f"  [Phase 7] gdn_scan: CUDA={cuda_ms:.4f}ms, PyTorch={py_ms:.4f}ms, Speedup={speedup:.2f}x")
    except Exception as e:
        results.append(
            {"phase": 7, "module": "gdn_scan", "cuda_ms": 0.0, "pytorch_ms": 0.0, "speedup": 0.0, "status": str(e)}
        )

    # Phase 9: gla_scan
    try:
        from neuralops.ops.cuda.gla_scan.gla_autograd import gla_scan, gla_scan_pytorch_ref

        q = torch.randn(2, 4, 512, 64, device=device, dtype=torch.float16)
        k = torch.randn(2, 4, 512, 64, device=device, dtype=torch.float16)
        v = torch.randn(2, 4, 512, 64, device=device, dtype=torch.float16)
        g = torch.sigmoid(torch.randn(2, 4, 512, device=device, dtype=torch.float16))
        cuda_ms = time_fn(gla_scan, q, k, v, g)
        py_ms = time_fn(gla_scan_pytorch_ref, q, k, v, g)
        speedup = py_ms / (cuda_ms + 1e-9)
        results.append(
            {
                "phase": 9,
                "module": "gla_scan",
                "cuda_ms": cuda_ms,
                "pytorch_ms": py_ms,
                "speedup": speedup,
                "status": "OK",
            }
        )
        print(f"  [Phase 9] gla_scan: CUDA={cuda_ms:.4f}ms, PyTorch={py_ms:.4f}ms, Speedup={speedup:.2f}x")
    except Exception as e:
        results.append(
            {"phase": 9, "module": "gla_scan", "cuda_ms": 0.0, "pytorch_ms": 0.0, "speedup": 0.0, "status": str(e)}
        )

    # Phase 9: selective_scan
    try:
        from neuralops.ops.cuda.selective_scan.selective_scan_autograd import selective_scan, selective_scan_pytorch_ref

        u = torch.randn(2, 64, 512, device=device, dtype=torch.float16)
        delta = torch.randn(2, 64, 512, device=device, dtype=torch.float16).abs() + 0.1
        A = -torch.rand(64, 16, device=device, dtype=torch.float16) - 0.1
        B = torch.randn(2, 16, 512, device=device, dtype=torch.float16)
        C = torch.randn(2, 16, 512, device=device, dtype=torch.float16)
        D = torch.randn(64, device=device, dtype=torch.float16)
        cuda_ms = time_fn(selective_scan, u, delta, A, B, C, D)
        py_ms = time_fn(selective_scan_pytorch_ref, u, delta, A, B, C, D)
        speedup = py_ms / (cuda_ms + 1e-9)
        results.append(
            {
                "phase": 9,
                "module": "selective_scan",
                "cuda_ms": cuda_ms,
                "pytorch_ms": py_ms,
                "speedup": speedup,
                "status": "OK",
            }
        )
        print(f"  [Phase 9] selective_scan: CUDA={cuda_ms:.4f}ms, PyTorch={py_ms:.4f}ms, Speedup={speedup:.2f}x")
    except Exception as e:
        results.append(
            {
                "phase": 9,
                "module": "selective_scan",
                "cuda_ms": 0.0,
                "pytorch_ms": 0.0,
                "speedup": 0.0,
                "status": str(e),
            }
        )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["phase", "module", "cuda_ms", "pytorch_ms", "speedup", "status"])
        writer.writeheader()
        writer.writerows(results)

    print(f"Speedup results saved to {args.output}")


if __name__ == "__main__":
    main()
