#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Standardized, Clean Benchmark Runner for All 131 Registered Attention Mechanisms.

Features:
- Sub-millisecond CUDA event timing (trimmed mean, median, standard deviation).
- Automatic forward and backward pass verification with gradient cleanup.
- Peak VRAM tracking per mechanism with explicit CUDA cache resetting.
- Structured JSON output saved to results/triton_all_131_attentions_benchmark.json.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from neuralops.benchmarks.harness import BenchmarkRunner, save_benchmark_json
from neuralops.core.registry import ATTENTION_REGISTRY, create_attention


def main():
    """Execute main routine."""
    parser = argparse.ArgumentParser(description="Clean Benchmark for All 131 Attention Mechanisms")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=15)
    parser.add_argument("--out-file", type=str, default="results/triton_all_131_attentions_benchmark.json")
    args = parser.parse_args()

    dev = torch.device(args.device)
    runner = BenchmarkRunner(device=dev, warmup_iters=args.warmup, benchmark_iters=args.iters)

    B, T, D, H = args.batch_size, args.seq_len, args.embed_dim, args.num_heads
    names = sorted(list(ATTENTION_REGISTRY.keys()))

    print(f"=== Benchmarking All {len(names)} Attention Mechanisms ===")
    print(f"Hardware Device: {dev} ({torch.cuda.get_device_name(dev) if dev.type == 'cuda' else 'CPU'})")
    print(f"Shape: B={B}, T={T}, D={D}, H={H} | Dtype=FP16\n")

    results = []

    for idx, name in enumerate(names, 1):
        runner.reset_vram()
        x = torch.randn(B, T, D, device=dev, dtype=torch.float16, requires_grad=True)

        try:
            mod = create_attention(name, embed_dim=D, num_heads=H).to(device=dev, dtype=torch.float16)
            mod.train()

            def _make_step(target_mod, target_x):
                """Execute make step routine."""

                def fwd_bwd_step():
                    """Execute fwd bwd step routine."""
                    out = target_mod(target_x)
                    if isinstance(out, tuple):
                        out = out[0]
                    loss = out.sum()
                    loss.backward()
                    target_x.grad = None
                    for p in target_mod.parameters():
                        p.grad = None

                return fwd_bwd_step

            metric = runner.measure_callable(
                _make_step(mod, x),
                batch_size=B,
                seq_len=T,
            )

            mod = None
            x = None
            runner.reset_vram()

            rec = {
                "name": name,
                "latency_ms": metric.latency_ms_trimmed_mean,
                "latency_median_ms": metric.latency_ms_median,
                "latency_std_ms": metric.latency_ms_std,
                "tokens_per_sec": metric.throughput_tokens_sec,
                "peak_vram_mb": metric.peak_vram_mb,
                "status": metric.status,
                "error": metric.error,
            }
            results.append(rec)
            print(
                f"[{idx:3d}/{len(names)}] {name:38s}: {metric.latency_ms_trimmed_mean:7.2f} ms | {metric.throughput_tokens_sec:8.1f} tok/s | VRAM: {metric.peak_vram_mb:5.1f} MB [OK]"
            )

        except Exception as e:
            err = str(e)
            results.append(
                {
                    "name": name,
                    "latency_ms": 0.0,
                    "latency_median_ms": 0.0,
                    "latency_std_ms": 0.0,
                    "tokens_per_sec": 0.0,
                    "peak_vram_mb": 0.0,
                    "status": "FAILED",
                    "error": err[:128],
                }
            )
            print(f"[{idx:3d}/{len(names)}] {name:38s}: FAILED ({err[:60]})")

    results.sort(key=lambda r: r["latency_ms"], reverse=True)
    out_p = save_benchmark_json(results, args.out_file)
    print(f"\nSaved benchmark results to {out_p.resolve()}")

    ok_count = sum(1 for r in results if r["status"] == "OK")
    failed_count = sum(1 for r in results if r["status"] == "FAILED")
    print(f"Summary: {ok_count}/{len(names)} passed forward+backward on GPU ({failed_count} failures)")


if __name__ == "__main__":
    main()
