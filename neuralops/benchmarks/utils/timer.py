#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Hardware-aware execution timer and statistical profiling utilities.

Complies strictly with AGENTS.md:
- Rule 2.2: CUDA timing via torch.cuda.Event and synchronize()
- Rule 2.3: L2 cache flush using device properties
- Rule 2.5: Safe device fallback for CPU and MPS
- Rule 9.1: Warmup iterations (>=10) before timing
- Rule 9.2: Outlier handling (median, p90, p99 after trimming top/bottom 5%)
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class TimingStats:
    """Summary statistics for benchmark latency measurements."""

    median_ms: float
    p90_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    runs_ms: tuple[float, ...]


def flush_l2(device: torch.device | None = None) -> None:
    """Flush the GPU L2 cache before a timed measurement per Rule 2.3.

    Args:
        device: Target CUDA device. If None, uses current active device.
    """
    if not torch.cuda.is_available():
        return
    try:
        dev = device if device is not None else torch.device("cuda")
        l2_bytes = torch.cuda.get_device_properties(dev).L2_cache_size
        if l2_bytes > 0:
            buffer = torch.empty(l2_bytes // 4, dtype=torch.int32, device=dev)
            buffer.zero_()
            torch.cuda.synchronize(dev)
    except Exception as exc:
        LOG.debug("L2 cache flush skipped or failed: %s", exc)


def compute_statistics(runs: list[float]) -> TimingStats:
    """Compute outlier-trimmed percentiles per Rule 9.2.

    Args:
        runs: List of execution times in milliseconds.

    Returns:
        TimingStats with median_ms, p90_ms, p99_ms, min_ms, max_ms, and runs_ms.
    """
    if not runs:
        return TimingStats(
            median_ms=0.0,
            p90_ms=0.0,
            p99_ms=0.0,
            min_ms=0.0,
            max_ms=0.0,
            runs_ms=(),
        )

    sorted_runs = sorted(runs)
    n = len(sorted_runs)

    # Discard top/bottom 5% if sample size allows
    trim = max(1, int(n * 0.05)) if n >= 20 else (1 if n >= 10 else 0)
    filtered = sorted_runs[trim : n - trim] if trim > 0 and (n - 2 * trim) > 0 else sorted_runs

    median_val = filtered[len(filtered) // 2]
    p90_idx = min(len(filtered) - 1, int(len(filtered) * 0.90))
    p99_idx = min(len(filtered) - 1, int(len(filtered) * 0.99))

    return TimingStats(
        median_ms=round(float(median_val), 4),
        p90_ms=round(float(filtered[p90_idx]), 4),
        p99_ms=round(float(filtered[p99_idx]), 4),
        min_ms=round(float(sorted_runs[0]), 4),
        max_ms=round(float(sorted_runs[-1]), 4),
        runs_ms=tuple(round(float(r), 4) for r in sorted_runs),
    )


def benchmark_callable(
    fn: Callable[[], Any],
    device: torch.device | str = "cpu",
    warmup_iters: int = 10,
    timed_iters: int = 25,
    flush_cache: bool = True,
) -> TimingStats:
    """Benchmark a callable with warmup and outlier trimming.

    Args:
        fn: Zero-argument callable to time.
        device: Device where computation occurs.
        warmup_iters: Minimum warmup iterations (default 10, per Rule 9.1).
        timed_iters: Number of iterations to measure.
        flush_cache: Whether to flush L2 cache before each run (CUDA only).

    Returns:
        TimingStats instance containing percentiles.
    """
    dev = torch.device(device) if isinstance(device, str) else device
    is_cuda = dev.type == "cuda" and torch.cuda.is_available()
    is_mps = dev.type == "mps" and hasattr(torch, "mps") and torch.backends.mps.is_available()

    # Rule 9.1: Warmup before timing
    for _ in range(max(10, warmup_iters)):
        fn()

    if is_cuda:
        torch.cuda.synchronize(dev)
    elif is_mps:
        torch.mps.synchronize()

    runs: list[float] = []

    for _ in range(timed_iters):
        if is_cuda:
            if flush_cache:
                flush_l2(dev)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            fn()
            end_event.record()
            torch.cuda.synchronize(dev)
            runs.append(start_event.elapsed_time(end_event))
        else:
            t0 = time.perf_counter()
            fn()
            if is_mps:
                torch.mps.synchronize()
            t1 = time.perf_counter()
            runs.append((t1 - t0) * 1000.0)

    return compute_statistics(runs)


__all__ = [
    "TimingStats",
    "flush_l2",
    "compute_statistics",
    "benchmark_callable",
]
