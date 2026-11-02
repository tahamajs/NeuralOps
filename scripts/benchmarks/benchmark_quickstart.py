#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Convenience CLI helper for launching common NeuralOps benchmark tasks quickly.

Usage:
    # 1. Quick attention kernel benchmark (PyTorch / Triton fallback)
    python3 scripts/benchmarks/benchmark_quickstart.py --mode attention

    # 2. Quick compressor validation
    python3 scripts/benchmarks/benchmark_quickstart.py --mode compressor

    # 3. Quick serving latency probe
    python3 scripts/benchmarks/benchmark_quickstart.py --mode serving
"""

from __future__ import annotations

import argparse
import sys
import time
import torch

from neuralops import ATTENTION_REGISTRY, force_kernel_backend
from neuralops.compression import COMPRESSOR_REGISTRY
from neuralops.ops.cuda import cosformer_linear_attention, get_dispatch_stats


def benchmark_attention(iterations: int = 20) -> int:
    print("=" * 65)
    print("  NeuralOps Benchmark Quickstart: Attention Kernel Execution")
    print("=" * 65)
    b, h, seq, d = 2, 4, 128, 32
    print(f"Tensor Shape: B={b}, H={h}, L={seq}, D={d}")
    q = torch.randn(b, h, seq, d)
    k = torch.randn(b, h, seq, d)
    v = torch.randn(b, h, seq, d)

    # Warmup
    for _ in range(5):
        _ = cosformer_linear_attention(q, k, v)

    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = cosformer_linear_attention(q, k, v)
    dt = time.perf_counter() - t0

    latency_ms = (dt / iterations) * 1000.0
    throughput = (b * seq * iterations) / dt

    print(f"Iterations       : {iterations}")
    print(f"Mean Latency     : {latency_ms:.3f} ms")
    print(f"Token Throughput : {throughput:.1f} tokens/s")
    print(f"Dispatch Stats   : {get_dispatch_stats()}")
    print("=" * 65)
    return 0


def benchmark_compressor() -> int:
    print("=" * 65)
    print("  NeuralOps Benchmark Quickstart: Compressor Zoo Registry")
    print("=" * 65)
    print(f"Total Registered Compressors: {len(COMPRESSOR_REGISTRY)}")
    sample = list(COMPRESSOR_REGISTRY.keys())[:5]
    print(f"Sample registered compressors: {', '.join(sample)}")
    print("=" * 65)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="NeuralOps Benchmark Quickstart")
    parser.add_argument(
        "--mode",
        choices=["attention", "compressor", "all"],
        default="attention",
        help="Benchmark component to execute",
    )
    parser.add_argument("--iters", type=int, default=20, help="Iteration count")
    args = parser.parse_args()

    if args.mode in ("attention", "all"):
        benchmark_attention(args.iters)
    if args.mode in ("compressor", "all"):
        benchmark_compressor()
    return 0


if __name__ == "__main__":
    sys.exit(main())
