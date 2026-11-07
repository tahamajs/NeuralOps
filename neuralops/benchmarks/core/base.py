#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Abstract base benchmark interface and execution summary dataclass.
"""

from __future__ import annotations

import logging
import math
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import torch

if TYPE_CHECKING:
    from neuralops.benchmarks.core.paths import RunDirectoryContext
    from neuralops.config.benchmark_schema import BenchmarkRunConfig
    from neuralops.tracking.tracker import ExperimentTracker

logger = logging.getLogger(__name__)


def _sanitize_float(val: Any, default: float = 0.0) -> float:
    """Ensure a numerical value is a finite float (not NaN or Inf)."""
    try:
        f = float(val)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


@dataclass
class ExecutionSummary:
    """Standardized summary of benchmark execution."""

    run_id: str
    benchmark_name: str
    status: str = "COMPLETED"  # "COMPLETED", "FAILED"
    total_iterations: int = 0
    passed_iterations: int = 0
    failed_oom_iterations: int = 0
    failed_runtime_iterations: int = 0
    mean_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p90_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    peak_vram_mb: float = 0.0
    mean_tflops: float = 0.0
    duration_seconds: float = 0.0
    results_table: List[Dict[str, Any]] = field(default_factory=list)
    extra_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert execution summary to serializable dictionary with sanitized floats."""
        d = asdict(self)
        for key in (
            "mean_latency_ms",
            "p50_latency_ms",
            "p90_latency_ms",
            "p99_latency_ms",
            "peak_vram_mb",
            "mean_tflops",
            "duration_seconds",
        ):
            d[key] = _sanitize_float(d.get(key, 0.0))
        return d


class AbstractBaseBenchmark(ABC):
    """Abstract base class governing all benchmark suites."""

    def __init__(
        self,
        config: BenchmarkRunConfig,
        tracker: ExperimentTracker,
        context: RunDirectoryContext,
    ) -> None:
        self.config = config
        self.tracker = tracker
        self.context = context
        self.device = self.resolve_device()
        self.results_table: List[Dict[str, Any]] = []

    def resolve_device(self) -> torch.device:
        """Resolve requested hardware execution device with graceful CPU fallback."""
        target = getattr(self.config.hardware, "device", "cuda:0")
        if target.startswith("cuda") and not torch.cuda.is_available():
            logger.warning(
                "CUDA requested ('%s') but torch.cuda.is_available() is False. Falling back to CPU.",
                target,
            )
            return torch.device("cpu")
        try:
            return torch.device(target)
        except Exception:
            return torch.device("cpu")

    def flush_l2_cache(self, size_mb: int = 40) -> None:
        """Flush GPU L2 cache by reading through an isolated memory buffer."""
        if self.device.type == "cuda" and torch.cuda.is_available():
            try:
                num_elements = (size_mb * 1024 * 1024) // 4
                dummy = torch.empty(num_elements, dtype=torch.float32, device=self.device)
                dummy.zero_()
                torch.cuda.synchronize(self.device)
                del dummy
            except Exception:
                pass

    def measure_latency(
        self,
        fn: Callable[[], Any],
        warmup: int = 3,
        repeat: int = 10,
    ) -> Tuple[float, List[float]]:
        """
        Measure execution latency using high-resolution PyTorch CUDA Events or CPU monotonic timers.
        Returns (mean_latency_ms, all_latencies_ms).
        """
        warmup = max(1, warmup)
        repeat = max(1, repeat)

        for _ in range(warmup):
            fn()

        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
            latencies: List[float] = []
            for _ in range(repeat):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                fn()
                end_event.record()
                end_event.synchronize()
                elapsed = _sanitize_float(start_event.elapsed_time(end_event))
                latencies.append(elapsed)
            mean_ms = sum(latencies) / len(latencies) if latencies else 0.0
            return _sanitize_float(mean_ms), latencies

        latencies_cpu: List[float] = []
        for _ in range(repeat):
            t0 = time.perf_counter()
            fn()
            t1 = time.perf_counter()
            elapsed_ms = _sanitize_float((t1 - t0) * 1000.0)
            latencies_cpu.append(elapsed_ms)
        mean_ms = sum(latencies_cpu) / len(latencies_cpu) if latencies_cpu else 0.0
        return _sanitize_float(mean_ms), latencies_cpu

    @abstractmethod
    def setup(self) -> None:
        """Initialize models, allocate buffers, and warm up GPU caches."""

    @abstractmethod
    def run_iteration(self, step: int, **kwargs: Any) -> Dict[str, float]:
        """Execute a single timed iteration with exact synchronization."""

    @abstractmethod
    def teardown(self) -> None:
        """Release GPU memory and finalize temporary resources."""

    @abstractmethod
    def generate_summary(self) -> ExecutionSummary:
        """Calculate descriptive metrics and format execution results."""

    @abstractmethod
    def run(self) -> None:
        """Execute all iterations across the parameter sweep matrix."""

    def execute(self) -> ExecutionSummary:
        """Template method orchestrating the complete benchmark lifecycle."""
        try:
            self.setup()
            self.run()
            summary = self.generate_summary()
            return summary
        finally:
            self.teardown()
