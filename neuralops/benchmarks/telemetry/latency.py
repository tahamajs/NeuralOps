#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Real-Time Latency Percentile Accumulator

Maintains running latency distributions and percentiles (P50, P90, P95, P99, P99.9)
with constant/bounded memory overhead.
"""

from __future__ import annotations

import math
import threading
from typing import Sequence


class LatencyDistributionTracker:
    """
    Accumulates latency samples and computes exact/reservoir percentiles and moments.
    Thread-safe for concurrent benchmark workers.
    """

    def __init__(self, max_samples: int = 100_000) -> None:
        self.max_samples = max_samples
        self._samples: list[float] = []
        self._count = 0
        self._sum = 0.0
        self._sum_sq = 0.0
        self._min = float("inf")
        self._max = float("-inf")
        self._lock = threading.Lock()

    def record(self, latency_ms: float) -> None:
        """Record an individual execution latency in milliseconds."""
        val = float(latency_ms)
        if not math.isfinite(val) or val < 0.0:
            return

        with self._lock:
            self._count += 1
            self._sum += val
            self._sum_sq += val * val
            if val < self._min:
                self._min = val
            if val > self._max:
                self._max = val

            if len(self._samples) < self.max_samples:
                self._samples.append(val)
            else:
                # Reservoir sampling replacement
                import random

                idx = random.randint(0, self._count - 1)
                if idx < self.max_samples:
                    self._samples[idx] = val

    def record_many(self, latencies: Sequence[float]) -> None:
        """Record multiple latency values at once."""
        for lat in latencies:
            self.record(lat)

    def percentile(self, p: float) -> float:
        """Compute the p-th percentile (0.0 <= p <= 100.0)."""
        with self._lock:
            if not self._samples:
                return 0.0
            sorted_samples = sorted(self._samples)
            n = len(sorted_samples)
            if n == 1:
                return sorted_samples[0]
            k = (n - 1) * (p / 100.0)
            f = math.floor(k)
            c = math.ceil(k)
            if f == c:
                return sorted_samples[int(k)]
            d0 = sorted_samples[int(f)] * (c - k)
            d1 = sorted_samples[int(c)] * (k - f)
            return d0 + d1

    def summary(self) -> dict[str, float]:
        """Return comprehensive distribution summary dictionary."""
        with self._lock:
            if self._count == 0:
                return {
                    "count": 0,
                    "mean_ms": 0.0,
                    "std_ms": 0.0,
                    "min_ms": 0.0,
                    "p50_ms": 0.0,
                    "p90_ms": 0.0,
                    "p95_ms": 0.0,
                    "p99_ms": 0.0,
                    "p99_9_ms": 0.0,
                    "max_ms": 0.0,
                }

            mean = self._sum / self._count
            variance = max(0.0, (self._sum_sq / self._count) - (mean * mean))
            std = math.sqrt(variance)

            # Extract percentiles from sorted sample buffer
            sorted_samples = sorted(self._samples)
            n = len(sorted_samples)

            def _pct(p: float) -> float:
                if n == 1:
                    return sorted_samples[0]
                k = (n - 1) * (p / 100.0)
                f = math.floor(k)
                c = math.ceil(k)
                if f == c:
                    return sorted_samples[int(k)]
                return sorted_samples[int(f)] * (c - k) + sorted_samples[int(c)] * (k - f)

            return {
                "count": self._count,
                "mean_ms": round(mean, 4),
                "std_ms": round(std, 4),
                "min_ms": round(self._min, 4),
                "p50_ms": round(_pct(50.0), 4),
                "p90_ms": round(_pct(90.0), 4),
                "p95_ms": round(_pct(95.0), 4),
                "p99_ms": round(_pct(99.0), 4),
                "p99_9_ms": round(_pct(99.9), 4),
                "max_ms": round(self._max, 4),
            }
