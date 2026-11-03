#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Canonical latency and VRAM scaling benchmark for attention-shaped inputs."""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from neuralops.benchmarks.config import BenchmarkConfig
from neuralops.benchmarks.results import BenchmarkResult, save_all_formats


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _attention_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    if hasattr(F, "scaled_dot_product_attention"):
        return F.scaled_dot_product_attention(q, k, v)
    scale = q.shape[-1] ** -0.5
    return torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1) @ v


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * percentile)))
    return ordered[index]


def run_resource_scaling(config: BenchmarkConfig) -> list[BenchmarkResult]:
    """Measure latency, throughput, and peak VRAM for configured shapes."""
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for resource scaling, but CUDA is unavailable.")
    device = torch.device("cuda" if config.device == "cuda" else config.device)
    dtype = getattr(torch, config.precision)
    output_dir = Path(config.output_dir)
    results: list[BenchmarkResult] = []

    for batch_size in config.batch_sizes:
        for seq_len in config.seq_lens:
            q = torch.randn(batch_size, 4, seq_len, 64, device=device, dtype=dtype)
            k = torch.randn_like(q)
            v = torch.randn_like(q)
            for _ in range(config.extra.get("warmup", 3)):
                _attention_reference(q, k, v)
            _synchronize(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            samples: list[float] = []
            for _ in range(config.extra.get("iters", 10)):
                start = time.perf_counter()
                output = _attention_reference(q, k, v)
                _synchronize(device)
                samples.append((time.perf_counter() - start) * 1000.0)
            if not torch.isfinite(output).all():
                raise RuntimeError(f"Non-finite attention output at batch={batch_size}, seq_len={seq_len}")
            peak_vram = torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0
            p50 = _percentile(samples, 0.50)
            p95 = _percentile(samples, 0.95)
            result = BenchmarkResult(
                module={"name": "scaled_dot_product_attention", "type": "kernel", "family": "attention"},
                model={"name": "synthetic_attention_probe"},
                dataset={"name": "synthetic", "task": "performance"},
                config={"device": str(device), "dtype": str(dtype), "batch_size": batch_size, "seq_len": seq_len},
                metrics={
                    "latency_ms": statistics.mean(samples),
                    "p50_ms": p50,
                    "p95_ms": p95,
                    "tokens_per_second": batch_size * seq_len / (statistics.mean(samples) / 1000.0),
                },
                resources={"peak_vram_mb": peak_vram, "warmup": config.extra.get("warmup", 3), "iterations": len(samples)},
                status="success",
            )
            save_all_formats(result, output_dir)
            results.append(result)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    config = BenchmarkConfig.from_file(args.config) if args.config else BenchmarkConfig()
    if args.device:
        config.device = args.device
    if args.output_dir:
        config.output_dir = args.output_dir
    run_resource_scaling(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
