#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
End-to-end serving and throughput benchmark suite.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import torch

from neuralops.benchmarks.core.base import AbstractBaseBenchmark, ExecutionSummary, _sanitize_float

logger = logging.getLogger(__name__)


class ServingBenchmark(AbstractBaseBenchmark):
    """
    Empirical benchmark suite measuring TTFT, inter-token latency, and serving throughput.
    """

    def setup(self) -> None:
        custom = self.config.custom_parameters or {}

        self.batch_sizes = custom.get("batch_sizes", [1, 4, 8])
        self.prompt_lens = custom.get("prompt_lens", [128, 512])
        self.gen_lens = custom.get("gen_lens", [32, 64])

        limit = custom.get("limit")
        if limit is not None and isinstance(limit, int) and limit > 0:
            self.batch_sizes = self.batch_sizes[:limit]

        self.start_time = time.time()
        self.results_table: List[Dict[str, Any]] = []

    def run_iteration(
        self,
        step: int,
        batch_size: int = 1,
        prompt_len: int = 128,
        gen_len: int = 32,
        **kwargs: Any,
    ) -> Dict[str, float]:
        raise NotImplementedError("e2e_serving requires real serving framework; not implemented")

    def run(self) -> None:
        step = 0
        p = self.prompt_lens[0] if self.prompt_lens else 128
        g = self.gen_lens[0] if self.gen_lens else 32
        for b in self.batch_sizes:
            self.run_iteration(step=step, batch_size=b, prompt_len=p, gen_len=g)
            step += 1

    def generate_summary(self) -> ExecutionSummary:
        duration = _sanitize_float(time.time() - getattr(self, "start_time", time.time()))
        total = len(self.results_table)
        passed = sum(1 for r in self.results_table if r.get("status") == "PASS")
        failed_runtime = total - passed

        passed_latencies = [
            r["latency_ms"] for r in self.results_table if r.get("status") == "PASS" and r.get("latency_ms", 0) > 0
        ]
        mean_lat = sum(passed_latencies) / len(passed_latencies) if passed_latencies else 0.0

        status = "COMPLETED" if (passed > 0 or total == 0) else "FAILED"

        return ExecutionSummary(
            run_id=self.context.run_id,
            benchmark_name="e2e_serving",
            status=status,
            total_iterations=total,
            passed_iterations=passed,
            failed_oom_iterations=0,
            failed_runtime_iterations=failed_runtime,
            mean_latency_ms=_sanitize_float(mean_lat),
            p50_latency_ms=_sanitize_float(mean_lat),
            p90_latency_ms=_sanitize_float(mean_lat),
            p99_latency_ms=_sanitize_float(mean_lat),
            peak_vram_mb=0.0,
            mean_tflops=0.0,
            duration_seconds=duration,
            results_table=self.results_table,
            extra_metadata={"batch_sizes_count": len(self.batch_sizes)},
        )

    def teardown(self) -> None:
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
