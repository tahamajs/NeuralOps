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
import math
import resource
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.benchmarks.config import BenchmarkConfig
from neuralops.benchmarks.results import BenchmarkResult, save_all_formats


def _is_cuda_device(device: str | torch.device) -> bool:
    dev_str = str(device).lower()
    return dev_str == "cuda" or dev_str.startswith("cuda:")


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _get_cpu_memory_mb() -> float:
    try:
        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return ru / (1024**2) if sys.platform == "darwin" else ru / 1024.0
    except Exception:
        return 0.0


def _attention_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    if hasattr(F, "scaled_dot_product_attention"):
        return F.scaled_dot_product_attention(q, k, v)
    scale = q.shape[-1] ** -0.5
    return torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1) @ v


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * percentile)))
    return ordered[index]


def _resolve_dtype(name: str) -> torch.dtype:
    aliases = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}
    canonical = aliases.get(name.lower(), name.lower())
    try:
        return getattr(torch, canonical)
    except AttributeError as exc:
        raise ValueError(f"Unsupported benchmark precision: {name}") from exc


def run_resource_scaling(config: BenchmarkConfig) -> list[BenchmarkResult]:
    """Measure latency, throughput, and peak VRAM for configured shapes."""
    if _is_cuda_device(config.device) and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for resource scaling, but CUDA is unavailable.")
    device = torch.device(config.device)
    dtype = _resolve_dtype(config.precision)
    output_dir = Path(config.output_dir)
    results: list[BenchmarkResult] = []

    attention_key = getattr(config, "attention_key", None)
    attention_module: nn.Module | None = None
    module_name = "scaled_dot_product_attention"

    if attention_key and attention_key != "scaled_dot_product_attention":
        from neuralops.core.registry import create_attention

        attention_module = create_attention(attention_key, dim=64, num_heads=4).to(device=device, dtype=dtype)
        attention_module.eval()
        module_name = attention_key

    for batch_size in config.batch_sizes:
        for seq_len in config.seq_lens:
            try:
                if attention_module is not None:
                    x = torch.randn(batch_size, seq_len, 64, device=device, dtype=dtype)

                    for _ in range(config.warmup):
                        with torch.no_grad():
                            _ = attention_module(x)
                    _synchronize(device)

                    if device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(device)
                    samples: list[float] = []
                    for _ in range(config.iters):
                        start = time.perf_counter()
                        with torch.no_grad():
                            out_raw = attention_module(x)
                        _synchronize(device)
                        samples.append((time.perf_counter() - start) * 1000.0)
                    output = out_raw[0] if isinstance(out_raw, tuple) else out_raw
                    del x, out_raw
                else:
                    q = torch.randn(batch_size, 4, seq_len, 64, device=device, dtype=dtype)
                    k = torch.randn_like(q)
                    v = torch.randn_like(q)
                    for _ in range(config.warmup):
                        _attention_reference(q, k, v)
                    _synchronize(device)
                    if device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(device)
                    samples = []
                    for _ in range(config.iters):
                        start = time.perf_counter()
                        output = _attention_reference(q, k, v)
                        _synchronize(device)
                        samples.append((time.perf_counter() - start) * 1000.0)
                    del q, k, v

                if not torch.isfinite(output).all():
                    raise RuntimeError(
                        f"FAIL_NUMERICS: Non-finite attention output at batch={batch_size}, seq_len={seq_len}"
                    )

                peak_vram = (
                    torch.cuda.max_memory_allocated(device) / (1024**2)
                    if device.type == "cuda"
                    else _get_cpu_memory_mb()
                )
                p50 = _percentile(samples, 0.50)
                p95 = _percentile(samples, 0.95)
                mean_latency = statistics.mean(samples)
                tokens_per_sec = (
                    batch_size * seq_len / (mean_latency / 1000.0) if mean_latency > 0 else 0.0
                )
                result = BenchmarkResult(
                    module={"name": module_name, "type": "kernel", "family": "attention"},
                    model={"name": "synthetic_attention_probe"},
                    dataset={"name": "synthetic", "task": "performance"},
                    config={"device": str(device), "dtype": str(dtype), "batch_size": batch_size, "seq_len": seq_len},
                    metrics={
                        "latency_ms": mean_latency,
                        "p50_ms": p50,
                        "p95_ms": p95,
                        "tokens_per_second": tokens_per_sec,
                    },
                    resources={"peak_vram_mb": peak_vram, "warmup": config.warmup, "iterations": len(samples)},
                    status="success",
                )
            except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                if "out of memory" in str(exc).lower():
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    result = BenchmarkResult(
                        module={"name": module_name, "type": "kernel", "family": "attention"},
                        model={"name": "synthetic_attention_probe"},
                        dataset={"name": "synthetic", "task": "performance"},
                        config={"device": str(device), "dtype": str(dtype), "batch_size": batch_size, "seq_len": seq_len},
                        metrics={
                            "latency_ms": float("nan"),
                            "p50_ms": float("nan"),
                            "p95_ms": float("nan"),
                            "tokens_per_second": 0.0,
                        },
                        resources={"peak_vram_mb": float("nan"), "warmup": config.warmup, "iterations": 0},
                        status="OOM",
                        error=str(exc),
                    )
                    save_all_formats(result, output_dir)
                    results.append(result)
                    break
                raise

            save_all_formats(result, output_dir)
            results.append(result)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None, help="Device to execute benchmark (cpu, cuda, cuda:0)")
    parser.add_argument("--attention", type=str, default=None, help="Attention mechanism registry key")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=None, help="Batch size list")
    parser.add_argument("--seq-lens", type=int, nargs="+", default=None, help="Sequence length list")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16", "float32", "float16", "bfloat16"], default=None)
    parser.add_argument("--warmup", type=int, default=None, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=None, help="Measurement iterations")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    config = BenchmarkConfig.from_file(args.config) if args.config else BenchmarkConfig()
    if args.device:
        config.device = args.device
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.batch_sizes:
        config.batch_sizes = args.batch_sizes
    if args.seq_lens:
        config.seq_lens = args.seq_lens
    if args.precision:
        config.precision = args.precision
    if args.warmup is not None:
        config.warmup = args.warmup
    if args.iters is not None:
        config.iters = args.iters
    if args.attention:
        config.attention_key = args.attention

    try:
        run_resource_scaling(config)
    except RuntimeError as exc:
        err_msg = str(exc)
        if "CUDA was requested for resource scaling, but CUDA is unavailable" in err_msg:
            sys.stderr.write(f"Error: {err_msg}\n")
            return 3  # AGENTS.md §10: 3 backend unavailable
        if "FAIL_NUMERICS" in err_msg:
            sys.stderr.write(f"Error: {err_msg}\n")
            return 4  # AGENTS.md §10: 4 numerical failure
        sys.stderr.write(f"Runtime error: {err_msg}\n")
        return 1
    except Exception as exc:
        sys.stderr.write(f"Unexpected error: {exc}\n")
        return 70
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
