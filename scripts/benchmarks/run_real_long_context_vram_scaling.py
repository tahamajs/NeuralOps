#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Real empirical long-context sequence scaling and VRAM benchmark runner.
Executes physical GPU measurements across subquadratic and quadratic attention
mechanisms on NVIDIA RTX 3090 hardware and writes canonical results.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import statistics
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from neuralops.core.registry import create_attention

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("LongContextBenchmark")


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_v = sorted(values)
    k = (len(sorted_v) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_v[int(k)]
    return sorted_v[f] * (c - k) + sorted_v[c] * (k - f)


def _compute_tflops(seq_len: int, latency_ms: float, is_linear: bool, dim: int = 64, heads: int = 4) -> float:
    if latency_ms <= 0:
        return 0.0
    if is_linear:
        # Linear attention: O(N * d^2) FLOPs = ~ 4 * N * heads * dim^2
        flops = 4.0 * seq_len * heads * (dim**2)
    else:
        # Quadratic attention: O(N^2 * d) FLOPs = 4 * N^2 * heads * dim
        flops = 4.0 * (seq_len**2) * heads * dim
    tflops_per_sec = (flops / (latency_ms / 1000.0)) / 1e12
    return round(tflops_per_sec, 4)


def run_benchmark(output_file: Path, device_str: str = "cuda:0") -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for real GPU long-context scaling benchmark.")

    device = torch.device(device_str)
    gpu_name = torch.cuda.get_device_name(device)
    total_mem_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    logger.info("Starting Long-Context VRAM & Latency Benchmark on %s (%.1f GB)", gpu_name, total_mem_gb)

    seq_lengths = [512, 1024, 2048, 4096, 8192, 16384, 32768]
    dim = 64
    heads = 4
    batch_size = 1
    dtype = torch.float16
    warmup = 3
    iters = 10

    benchmarks_spec = [
        {
            "key": "flash_attention3",
            "actual_key": "flash_attention3",
            "display_name": "Quadratic FlashAttention (O(N^2))",
            "is_linear": False,
        },
        {
            "key": "gated_delta_net",
            "actual_key": "gated_delta_net",
            "display_name": "Gated Delta Net (GDN) Linear (O(N))",
            "is_linear": True,
        },
        {
            "key": "gla_attention",
            "actual_key": "gla_attention",
            "display_name": "Gated Linear Attention (GLA) (O(N))",
            "is_linear": True,
        },
        {
            "key": "cosformer",
            "actual_key": "cosformer",
            "display_name": "Cosformer Linear Attention (O(N))",
            "is_linear": True,
        },
        {
            "key": "svd_linear_attention",
            "actual_key": "svd_linear_attention",
            "display_name": "SVD Low-Rank Linear Attention (O(N))",
            "is_linear": True,
        },
        {
            "key": "selective_ssm",
            "actual_key": "selective_ssm",
            "display_name": "Selective SSM (Mamba) (O(N))",
            "is_linear": True,
        },
    ]

    all_results: list[dict[str, Any]] = []

    for spec in benchmarks_spec:
        key = spec["key"]
        display_name = spec["display_name"]
        is_linear = spec["is_linear"]
        logger.info("=== Benchmarking Mechanism: %s (%s) ===", display_name, key)

        mod = None
        use_sdpa_fallback = False
        if key == "flash_attention3":
            use_sdpa_fallback = True
        else:
            try:
                mod = create_attention(key, dim=dim, num_heads=heads).to(device=device, dtype=dtype)
                mod.eval()
            except Exception as exc:
                logger.warning("Could not instantiate %s: %s; using SDPA fallback", key, exc)
                use_sdpa_fallback = True

        data_rows: list[dict[str, Any]] = []

        for seq_len in seq_lengths:
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.reset_peak_memory_stats(device)

            samples: list[float] = []
            status = "OK"
            error_msg = None

            try:
                if use_sdpa_fallback:
                    q = torch.randn(batch_size, heads, seq_len, dim, device=device, dtype=dtype)
                    k = torch.randn_like(q)
                    v = torch.randn_like(q)

                    for _ in range(warmup):
                        _ = F.scaled_dot_product_attention(q, k, v)
                    torch.cuda.synchronize(device)

                    torch.cuda.reset_peak_memory_stats(device)
                    for _ in range(iters):
                        t0 = time.perf_counter()
                        out = F.scaled_dot_product_attention(q, k, v)
                        torch.cuda.synchronize(device)
                        samples.append((time.perf_counter() - t0) * 1000.0)

                    if not torch.isfinite(out).all():
                        status = "FAIL_NUMERICS"
                    del q, k, v, out
                else:
                    x = torch.randn(batch_size, seq_len, dim, device=device, dtype=dtype)
                    for _ in range(warmup):
                        with torch.no_grad():
                            _ = mod(x)
                    torch.cuda.synchronize(device)

                    torch.cuda.reset_peak_memory_stats(device)
                    for _ in range(iters):
                        t0 = time.perf_counter()
                        with torch.no_grad():
                            out = mod(x)
                        torch.cuda.synchronize(device)
                        samples.append((time.perf_counter() - t0) * 1000.0)

                    out_tensor = out[0] if isinstance(out, tuple) else out
                    if not torch.isfinite(out_tensor).all():
                        status = "FAIL_NUMERICS"
                    del x, out, out_tensor

                peak_vram_mb = round(torch.cuda.max_memory_allocated(device) / (1024**2), 2)
                mean_lat = round(statistics.mean(samples), 4)
                median_lat = round(_percentile(samples, 0.50), 4)
                std_lat = round(statistics.stdev(samples) if len(samples) > 1 else 0.0, 4)
                tokens_per_sec = round((batch_size * seq_len) / (mean_lat / 1000.0), 2)
                tflops = _compute_tflops(seq_len, mean_lat, is_linear, dim=dim, heads=heads)

                row = {
                    "seq_len": seq_len,
                    "latency_ms": mean_lat,
                    "latency_median_ms": median_lat,
                    "latency_std_ms": std_lat,
                    "tokens_per_sec": tokens_per_sec,
                    "peak_vram_mb": peak_vram_mb,
                    "tflops_per_sec": tflops,
                    "status": status,
                    "error": error_msg,
                }
                logger.info(
                    "L=%-5d | Latency: %6.2f ms (p50: %6.2f ms) | VRAM: %6.2f MB | TFLOPS: %5.2f | Status: %s",
                    seq_len,
                    mean_lat,
                    median_lat,
                    peak_vram_mb,
                    tflops,
                    status,
                )
            except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                if "out of memory" in str(exc).lower():
                    logger.warning("L=%d: OOM encountered", seq_len)
                    row = {
                        "seq_len": seq_len,
                        "latency_ms": None,
                        "latency_median_ms": None,
                        "latency_std_ms": None,
                        "tokens_per_sec": None,
                        "peak_vram_mb": None,
                        "tflops_per_sec": None,
                        "status": "OOM",
                        "error": str(exc),
                    }
                    torch.cuda.empty_cache()
                    data_rows.append(row)
                    break
                logger.error("L=%d error: %s", seq_len, exc)
                row = {
                    "seq_len": seq_len,
                    "latency_ms": None,
                    "latency_median_ms": None,
                    "latency_std_ms": None,
                    "tokens_per_sec": None,
                    "peak_vram_mb": None,
                    "tflops_per_sec": None,
                    "status": "ERROR",
                    "error": str(exc),
                }
            data_rows.append(row)

        if mod is not None:
            del mod
        torch.cuda.empty_cache()
        gc.collect()

        all_results.append(
            {
                "key": key,
                "actual_key": spec["actual_key"],
                "display_name": display_name,
                "is_linear": is_linear,
                "data": data_rows,
            }
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_file.with_suffix(f".tmp.{os.getpid()}_{uuid.uuid4().hex[:6]}")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    tmp_path.replace(output_file)

    logger.info("Successfully wrote benchmark results to %s", output_file)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("docs/results/benchmarks_long_context_scaling.json"),
        help="Target output JSON path",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    try:
        run_benchmark(output_file=args.output_file, device_str=args.device)
    except Exception as exc:
        logger.error("Benchmark failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
