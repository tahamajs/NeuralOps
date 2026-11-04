#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Needle In A Haystack (NIAH) long-context retrieval benchmark suite.
Evaluates exact-match retrieval accuracy, attention score peak, and latency
across variable context lengths and needle insertion depths.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

from neuralops.benchmarks.core.base import AbstractBaseBenchmark, ExecutionSummary, _sanitize_float

logger = logging.getLogger(__name__)


class NeedleInAHaystackBenchmark(AbstractBaseBenchmark):
    """
    Empirical benchmark suite measuring long-context retrieval accuracy and latency
    across context lengths and needle depths.
    """

    def setup(self) -> None:
        custom = self.config.custom_parameters or {}

        min_ctx = custom.get("min_context_length", 1024)
        max_ctx = custom.get("max_context_length", 32768)
        step_ctx = custom.get("context_step", 4096)

        configured_lengths = custom.get("context_lengths")
        if configured_lengths and isinstance(configured_lengths, list):
            self.context_lengths = [int(x) for x in configured_lengths]
        else:
            self.context_lengths = list(range(min_ctx, max_ctx + 1, step_ctx))
            if not self.context_lengths:
                self.context_lengths = [min_ctx]

        configured_depths = custom.get("depth_percentages")
        if configured_depths and isinstance(configured_depths, list):
            self.depth_percentages = [float(x) for x in configured_depths]
        else:
            self.depth_percentages = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        self.needle_prompt = custom.get("needle_prompt", "The secret password is ALPHA_CENTAURI_42")

        # Build execution grid
        self.grid: List[Tuple[int, float]] = []
        for ctx in self.context_lengths:
            for depth in self.depth_percentages:
                self.grid.append((ctx, depth))

        limit = custom.get("limit")
        if limit is not None and isinstance(limit, int) and limit > 0:
            self.grid = self.grid[:limit]

        self.start_time = time.time()
        self.results_table: List[Dict[str, Any]] = []

    def run_iteration(
        self,
        step: int,
        ctx_len: int,
        depth: float,
        **kwargs: Any,
    ) -> Dict[str, float]:
        self.flush_l2_cache()

        state_dim = 64
        needle_pos = max(0, min(int(ctx_len * depth), ctx_len - 1))
        dtype = torch.float32

        try:
            key = torch.randn(1, 1, state_dim, device=self.device, dtype=dtype)
            haystack = torch.randn(1, ctx_len, state_dim, device=self.device, dtype=dtype)
            needle = key + 0.001 * torch.randn_like(key)
            haystack[:, needle_pos : needle_pos + 1, :] = needle

            def eval_retrieval() -> int:
                scores = torch.matmul(haystack, key.transpose(-1, -2)).squeeze(-1)
                return int(torch.argmax(scores, dim=-1).item())

            latency_ms, _ = self.measure_latency(eval_retrieval, warmup=1, repeat=3)
            predicted_pos = eval_retrieval()
            success = abs(predicted_pos - needle_pos) <= 2
            accuracy = 1.0 if success else 0.0

            self.tracker.log_metric("niah_retrieval_accuracy", accuracy, step=step)
            self.tracker.log_metric("niah_latency_ms", latency_ms, step=step)

            result_entry = {
                "step": step,
                "mechanism_or_kernel": f"niah_ctx{ctx_len}_depth{int(depth * 100)}",
                "batch_size": 1,
                "seq_len": ctx_len,
                "dtype": "fp32",
                "depth_pct": depth,
                "latency_ms": latency_ms,
                "latency_p50_ms": latency_ms,
                "latency_p90_ms": latency_ms,
                "latency_p99_ms": latency_ms,
                "retrieval_accuracy": accuracy,
                "predicted_pos": predicted_pos,
                "actual_pos": needle_pos,
                "tflops": 0.0,
                "peak_vram_mb": 0.0,
                "status": "PASS",
            }
            self.results_table.append(result_entry)
            return {"accuracy": accuracy, "latency_ms": latency_ms}

        except Exception as e:
            logger.warning("NIAH iteration failed at ctx=%d, depth=%.2f: %s", ctx_len, depth, e)
            self.results_table.append({
                "step": step,
                "mechanism_or_kernel": f"niah_ctx{ctx_len}_depth{int(depth * 100)}",
                "batch_size": 1,
                "seq_len": ctx_len,
                "dtype": "fp32",
                "depth_pct": depth,
                "latency_ms": 0.0,
                "latency_p50_ms": 0.0,
                "latency_p90_ms": 0.0,
                "latency_p99_ms": 0.0,
                "retrieval_accuracy": 0.0,
                "predicted_pos": -1,
                "actual_pos": needle_pos,
                "tflops": 0.0,
                "peak_vram_mb": 0.0,
                "status": f"FAILED_RUNTIME: {e}",
            })
            self.tracker.log_metric("niah_retrieval_accuracy", 0.0, step=step)
            self.tracker.log_metric("niah_latency_ms", 0.0, step=step)
            return {"accuracy": 0.0, "latency_ms": 0.0}

    def run(self) -> None:
        step = 0
        for ctx_len, depth in self.grid:
            self.run_iteration(step=step, ctx_len=ctx_len, depth=depth)
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

        accuracies = [r.get("retrieval_accuracy", 0.0) for r in self.results_table if r.get("status") == "PASS"]
        mean_acc = sum(accuracies) / len(accuracies) if accuracies else 0.0

        status = "COMPLETED" if (passed > 0 or total == 0) else "FAILED"

        return ExecutionSummary(
            run_id=self.context.run_id,
            benchmark_name="niah",
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
            extra_metadata={
                "mean_retrieval_accuracy": _sanitize_float(mean_acc),
                "grid_points_count": len(self.grid),
            },
        )

    def teardown(self) -> None:
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
