#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unified High-Precision Benchmark Base Harness for NeuralOps

Implements harness functionality for NeuralOps.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import gc
import json
from pathlib import Path
import time
from typing import Any, Callable, Dict, Optional, Union

import numpy as np
import torch


@dataclass
class HardwareProfile:
    """Hardware specification constants for RTX 4070 Laptop / Ampere / Ada Lovelace."""
    name: str = "NVIDIA GeForce RTX 4070 Laptop GPU"
    peak_bandwidth_gb_s: float = 264.0
    peak_fp16_tflops: float = 79.0
    peak_bf16_tflops: float = 79.0
    peak_fp32_tflops: float = 20.0
    vram_total_mb: float = 8188.0


@dataclass
class BenchmarkMetric:
    """Detailed statistical measurements of a benchmark run."""
    latency_ms_mean: float = 0.0
    latency_ms_median: float = 0.0
    latency_ms_std: float = 0.0
    latency_ms_trimmed_mean: float = 0.0
    latency_ms_min: float = 0.0
    latency_ms_max: float = 0.0
    throughput_tokens_sec: float = 0.0
    throughput_samples_sec: float = 0.0
    peak_vram_mb: float = 0.0
    allocated_vram_mb: float = 0.0
    tflops_per_sec: float = 0.0
    bandwidth_gb_s: float = 0.0
    bandwidth_utilization_pct: float = 0.0
    arithmetic_intensity: float = 0.0
    status: str = "OK"
    error: Optional[str] = None
    extra_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BenchmarkRunner:
    """Robust, reproducible GPU timing & profiling harness."""

    def __init__(
        self,
        device: Optional[Union[str, torch.device]] = None,
        warmup_iters: int = 5,
        benchmark_iters: int = 25,
        hardware_profile: Optional[HardwareProfile] = None,
    ) -> None:
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.warmup_iters = warmup_iters
        self.benchmark_iters = benchmark_iters
        self.hw = hardware_profile or HardwareProfile()

    def reset_vram(self) -> None:
        """Force garbage collection and reset CUDA memory peak trackers."""
        gc.collect()
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)

    def get_peak_vram_mb(self) -> float:
        """Return peak allocated VRAM in MB since last reset."""
        if self.device.type == "cuda" and torch.cuda.is_available():
            return torch.cuda.max_memory_allocated(self.device) / (1024.0 * 1024.0)
        return 0.0

    def get_allocated_vram_mb(self) -> float:
        """Return currently allocated VRAM in MB."""
        if self.device.type == "cuda" and torch.cuda.is_available():
            return torch.cuda.memory_allocated(self.device) / (1024.0 * 1024.0)
        return 0.0

    def measure_callable(
        self,
        fn: Callable[[], Any],
        total_flops: float = 0.0,
        total_bytes: float = 0.0,
        batch_size: int = 1,
        seq_len: int = 1,
        warmup: Optional[int] = None,
        iters: Optional[int] = None,
    ) -> BenchmarkMetric:
        """Measure execution time with CUDA events, peak VRAM, FLOPs, and bandwidth.

        Uses CUDA events when on CUDA device for sub-millisecond precision,
        synchronized before and after every recorded iteration.
        """
        w_iters = warmup if warmup is not None else self.warmup_iters
        b_iters = iters if iters is not None else self.benchmark_iters

        self.reset_vram()

        try:
            for _ in range(w_iters):
                fn()

            if self.device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize(self.device)
                self.reset_vram()

                start_events = [torch.cuda.Event(enable_timing=True) for _ in range(b_iters)]
                end_events = [torch.cuda.Event(enable_timing=True) for _ in range(b_iters)]

                for i in range(b_iters):
                    start_events[i].record()
                    fn()
                    end_events[i].record()

                torch.cuda.synchronize(self.device)
                latencies = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
            else:
                latencies = []
                for _ in range(b_iters):
                    t0 = time.perf_counter()
                    fn()
                    t1 = time.perf_counter()
                    latencies.append((t1 - t0) * 1000.0)

            lat_arr = np.array(latencies, dtype=np.float64)
            mean_lat = float(np.mean(lat_arr))
            median_lat = float(np.median(lat_arr))
            std_lat = float(np.std(lat_arr))
            min_lat = float(np.min(lat_arr))
            max_lat = float(np.max(lat_arr))

            if len(lat_arr) >= 5:
                k = max(1, int(len(lat_arr) * 0.1))
                sorted_lats = np.sort(lat_arr)
                trimmed_lat = float(np.mean(sorted_lats[k:-k]))
            else:
                trimmed_lat = mean_lat

            peak_vram = self.get_peak_vram_mb()
            alloc_vram = self.get_allocated_vram_mb()

            total_tokens = batch_size * seq_len
            throughput_tokens = (total_tokens / (trimmed_lat / 1000.0)) if trimmed_lat > 0 else 0.0
            throughput_samples = (batch_size / (trimmed_lat / 1000.0)) if trimmed_lat > 0 else 0.0

            latency_s = trimmed_lat / 1000.0
            tflops_s = (total_flops / (latency_s * 1e12)) if (total_flops > 0 and latency_s > 0) else 0.0
            bandwidth_gb_s = (total_bytes / (latency_s * 1e9)) if (total_bytes > 0 and latency_s > 0) else 0.0
            bandwidth_pct = (bandwidth_gb_s / self.hw.peak_bandwidth_gb_s * 100.0) if self.hw.peak_bandwidth_gb_s > 0 else 0.0
            arithmetic_intensity = (total_flops / total_bytes) if total_bytes > 0 else 0.0

            return BenchmarkMetric(
                latency_ms_mean=round(mean_lat, 4),
                latency_ms_median=round(median_lat, 4),
                latency_ms_std=round(std_lat, 4),
                latency_ms_trimmed_mean=round(trimmed_lat, 4),
                latency_ms_min=round(min_lat, 4),
                latency_ms_max=round(max_lat, 4),
                throughput_tokens_sec=round(throughput_tokens, 2),
                throughput_samples_sec=round(throughput_samples, 2),
                peak_vram_mb=round(peak_vram, 2),
                allocated_vram_mb=round(alloc_vram, 2),
                tflops_per_sec=round(tflops_s, 3),
                bandwidth_gb_s=round(bandwidth_gb_s, 2),
                bandwidth_utilization_pct=round(bandwidth_pct, 2),
                arithmetic_intensity=round(arithmetic_intensity, 2),
                status="OK",
                error=None,
            )

        except Exception as e:
            err = str(e)
            return BenchmarkMetric(
                status="FAILED",
                error=err[:256],
            )


# Backward-compatible public name retained for callers that select the
# multi-mode benchmark through the top-level NeuralOps API.
MultiModeBenchmarkRunner = BenchmarkRunner


def save_benchmark_json(data: Any, out_path: Union[str, Path]) -> Path:
    """Save benchmark results to structured JSON with atomic write."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    temp_path.replace(path)
    return path
