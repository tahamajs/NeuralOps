#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Analytical and empirical hardware roofline benchmark suite.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import torch

from neuralops.benchmarks.core.base import AbstractBaseBenchmark, ExecutionSummary, _sanitize_float
from neuralops.benchmarks.roofline import roofline_metrics

logger = logging.getLogger(__name__)


class RooflineBenchmark(AbstractBaseBenchmark):
    """
    Empirical benchmark suite measuring arithmetic intensity and hardware roofline saturation.
    """

    def setup(self) -> None:
        custom = self.config.custom_parameters or {}

        self.attention_types = custom.get("attention_types", ["full", "causal", "mla"])
        self.batch_sizes = custom.get("batch_sizes", [1, 2])
        self.seq_lens = custom.get("seq_lens", [128, 256])
        self.head_dim = custom.get("head_dim", 64)
        self.num_heads = custom.get("num_heads", 4)

        limit = custom.get("limit")
        if limit is not None and isinstance(limit, int) and limit > 0:
            self.attention_types = self.attention_types[:limit]

        self.start_time = time.time()
        self.results_table: List[Dict[str, Any]] = []

    def run_iteration(
        self,
        step: int,
        attention_type: str,
        batch_size: int = 1,
        seq_len: int = 128,
        **kwargs: Any,
    ) -> Dict[str, float]:
        self.flush_l2_cache()

        # Dummy execution function simulating attention
        def forward_fn() -> None:
            q = torch.randn(batch_size, self.num_heads, seq_len, self.head_dim, device=self.device)
            k = torch.randn(batch_size, self.num_heads, seq_len, self.head_dim, device=self.device)
            _ = torch.matmul(q, k.transpose(-2, -1))

        try:
            latency_ms, _ = self.measure_latency(forward_fn, warmup=1, repeat=3)

            metrics = roofline_metrics(
                attention_type=attention_type,
                B=batch_size,
                H=self.num_heads,
                S=seq_len,
                D=self.head_dim,
                latency_ms=latency_ms,
            )

            achieved_tflops = _sanitize_float(metrics.get("achieved_tflops", 0.0))
            arithmetic_intensity = _sanitize_float(metrics.get("arithmetic_intensity", 0.0))

            self.tracker.log_metric("achieved_tflops", achieved_tflops, step=step)

            result_entry = {
                "step": step,
                "mechanism_or_kernel": f"roofline_{attention_type}",
                "batch_size": batch_size,
                "seq_len": seq_len,
                "dtype": "fp32",
                "latency_ms": latency_ms,
                "latency_p50_ms": latency_ms,
                "latency_p90_ms": latency_ms,
                "latency_p99_ms": latency_ms,
                "tflops": achieved_tflops,
                "arithmetic_intensity": arithmetic_intensity,
                "peak_vram_mb": 0.0,
                "status": "PASS",
            }
            self.results_table.append(result_entry)
            return {"achieved_tflops": achieved_tflops, "arithmetic_intensity": arithmetic_intensity}

        except Exception as e:
            logger.warning("Roofline calculation error: %s", e)
            self.results_table.append({
                "step": step,
                "mechanism_or_kernel": f"roofline_{attention_type}",
                "batch_size": batch_size,
                "seq_len": seq_len,
                "dtype": "fp32",
                "latency_ms": 0.0,
                "latency_p50_ms": 0.0,
                "latency_p90_ms": 0.0,
                "latency_p99_ms": 0.0,
                "tflops": 0.0,
                "arithmetic_intensity": 0.0,
                "peak_vram_mb": 0.0,
                "status": f"FAILED_RUNTIME: {e}",
            })
            self.tracker.log_metric("achieved_tflops", 0.0, step=step)
            return {"achieved_tflops": 0.0, "arithmetic_intensity": 0.0}

    def run(self) -> None:
        step = 0
        b = self.batch_sizes[0] if self.batch_sizes else 1
        s = self.seq_lens[0] if self.seq_lens else 128
        for attn in self.attention_types:
            self.run_iteration(step=step, attention_type=attn, batch_size=b, seq_len=s)
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
        tflops_list = [r.get("tflops", 0.0) for r in self.results_table if r.get("status") == "PASS"]
        mean_tflops = sum(tflops_list) / len(tflops_list) if tflops_list else 0.0

        status = "COMPLETED" if (passed > 0 or total == 0) else "FAILED"

        return ExecutionSummary(
            run_id=self.context.run_id,
            benchmark_name="roofline",
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
            mean_tflops=_sanitize_float(mean_tflops),
            duration_seconds=duration,
            results_table=self.results_table,
            extra_metadata={"attention_types_count": len(self.attention_types)},
        )

    def teardown(self) -> None:
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
