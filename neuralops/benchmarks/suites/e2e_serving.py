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
        self.flush_l2_cache()

        dim = 64

        # Prefill simulation
        def prefill() -> Any:
            x = torch.randn(batch_size, prompt_len, dim, device=self.device)
            return torch.matmul(x, x.transpose(-1, -2))

        # Decode simulation
        def decode_step() -> Any:
            t = torch.randn(batch_size, 1, dim, device=self.device)
            return t * 2.0

        try:
            ttft_ms, _ = self.measure_latency(prefill, warmup=1, repeat=2)
            itl_ms, _ = self.measure_latency(decode_step, warmup=1, repeat=gen_len)

            total_tokens = batch_size * gen_len
            total_time_s = (ttft_ms + itl_ms * gen_len) / 1000.0
            throughput = _sanitize_float(total_tokens / max(total_time_s, 1e-6))

            self.tracker.log_metric("ttft_ms", ttft_ms, step=step)
            self.tracker.log_metric("throughput_tokens_per_sec", throughput, step=step)

            result_entry = {
                "step": step,
                "mechanism_or_kernel": "serving_simulation",
                "batch_size": batch_size,
                "seq_len": prompt_len + gen_len,
                "dtype": "fp32",
                "latency_ms": ttft_ms,
                "latency_p50_ms": ttft_ms,
                "latency_p90_ms": ttft_ms,
                "latency_p99_ms": ttft_ms,
                "ttft_ms": ttft_ms,
                "itl_ms": itl_ms,
                "throughput_tokens_per_sec": throughput,
                "tflops": 0.0,
                "peak_vram_mb": 0.0,
                "status": "PASS",
            }
            self.results_table.append(result_entry)
            return {"ttft_ms": ttft_ms, "throughput_tokens_per_sec": throughput}

        except Exception as e:
            logger.warning("Serving simulation error: %s", e)
            self.results_table.append(
                {
                    "step": step,
                    "mechanism_or_kernel": "serving_simulation",
                    "batch_size": batch_size,
                    "seq_len": prompt_len + gen_len,
                    "dtype": "fp32",
                    "latency_ms": 0.0,
                    "latency_p50_ms": 0.0,
                    "latency_p90_ms": 0.0,
                    "latency_p99_ms": 0.0,
                    "ttft_ms": 0.0,
                    "itl_ms": 0.0,
                    "throughput_tokens_per_sec": 0.0,
                    "tflops": 0.0,
                    "peak_vram_mb": 0.0,
                    "status": f"FAILED_RUNTIME: {e}",
                }
            )
            self.tracker.log_metric("ttft_ms", 0.0, step=step)
            return {"ttft_ms": 0.0, "throughput_tokens_per_sec": 0.0}

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
