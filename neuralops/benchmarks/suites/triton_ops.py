#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Triton ops benchmark suite for NeuralOps JIT kernel modules.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from neuralops.benchmarks.core.base import AbstractBaseBenchmark, ExecutionSummary, _sanitize_float
from neuralops.ops.triton._common import (
    gqa_pytorch_fallback,
    linear_attn_pytorch_fallback,
    low_rank_pytorch_fallback,
    sparse_attn_pytorch_fallback,
    state_update_pytorch_fallback,
    triton_dispatch,
)

logger = logging.getLogger(__name__)


class TritonOpsBenchmark(AbstractBaseBenchmark):
    def setup(self) -> None:
        custom = self.config.custom_parameters or {}

        all_ops = [
            "linear_attn",
            "low_rank",
            "sparse_attn",
            "gqa",
            "state_update",
        ]

        cfg_ops = custom.get("kernels")
        if cfg_ops:
            self.target_kernels = list(cfg_ops)
        else:
            self.target_kernels = all_ops

        limit = custom.get("limit")
        if limit is not None and isinstance(limit, int) and limit > 0:
            self.target_kernels = self.target_kernels[:limit]

        self.warmup_iterations = custom.get("warmup_iterations", 3)
        self.benchmark_iterations = custom.get("benchmark_iterations", 10)

        self.start_time = time.time()
        self.results_table: List[Dict[str, Any]] = []

    def _prepare_inputs(self, kernel_name: str) -> Tuple[Tuple[Any, ...], Callable[..., Any]]:
        dtype = torch.float16 if (self.device.type == "cuda" and torch.cuda.is_available()) else torch.float32

        if kernel_name == "linear_attn":
            q = torch.randn(2, 4, 16, 32, device=self.device, dtype=dtype)
            k = torch.randn(2, 4, 16, 32, device=self.device, dtype=dtype)
            v = torch.randn(2, 4, 16, 32, device=self.device, dtype=dtype)
            return (q, k, v), linear_attn_pytorch_fallback

        elif kernel_name == "low_rank":
            x = torch.randn(2, 16, 64, device=self.device, dtype=dtype)
            U = torch.randn(16, 64, device=self.device, dtype=dtype)
            V = torch.randn(16, 64, device=self.device, dtype=dtype)
            return (x, U, V), low_rank_pytorch_fallback

        elif kernel_name == "sparse_attn":
            q = torch.randn(2, 4, 16, 32, device=self.device, dtype=dtype)
            k = torch.randn(2, 4, 16, 32, device=self.device, dtype=dtype)
            v = torch.randn(2, 4, 16, 32, device=self.device, dtype=dtype)
            return (q, k, v), sparse_attn_pytorch_fallback

        elif kernel_name == "gqa":
            k = torch.randn(2, 2, 16, 32, device=self.device, dtype=dtype)
            v = torch.randn(2, 2, 16, 32, device=self.device, dtype=dtype)
            return (k, v, 2, 8), gqa_pytorch_fallback

        elif kernel_name == "state_update":
            h = torch.randn(2, 4, 32, 32, device=self.device, dtype=dtype)
            x = torch.randn(2, 4, 16, 32, device=self.device, dtype=dtype)
            return (h, x), state_update_pytorch_fallback

        else:
            x = torch.randn(2, 16, 32, device=self.device, dtype=dtype)
            return (x,), lambda t: t * 2.0

    def run_iteration(
        self,
        step: int,
        kernel_name: str,
        **kwargs: Any,
    ) -> Dict[str, float]:
        self.flush_l2_cache()

        args, ref_fn = self._prepare_inputs(kernel_name)

        try:

            def dispatched_call() -> Any:
                return triton_dispatch(kernel_name, *args)

            dispatched_ms, _ = self.measure_latency(
                dispatched_call,
                warmup=self.warmup_iterations,
                repeat=self.benchmark_iterations,
            )

            def ref_call() -> Any:
                return ref_fn(*args)

            ref_ms, _ = self.measure_latency(
                ref_call,
                warmup=self.warmup_iterations,
                repeat=self.benchmark_iterations,
            )

            out_disp = dispatched_call()
            out_ref = ref_call()
            if isinstance(out_disp, tuple):
                out_disp = out_disp[0]
            if isinstance(out_ref, tuple):
                out_ref = out_ref[0]

            is_finite = bool(torch.isfinite(out_disp).all().item()) and bool(torch.isfinite(out_ref).all().item())
            diff = (out_disp.float() - out_ref.float()).abs().max().item() if is_finite else float("inf")
            max_diff = _sanitize_float(diff)

            speedup = _sanitize_float(ref_ms / max(dispatched_ms, 1e-6))
            status = "PASS" if is_finite and max_diff < 1e-2 else "FAIL_NUMERICS"

            self.tracker.log_metric("triton_ms", dispatched_ms, step=step)

            result_entry = {
                "step": step,
                "mechanism_or_kernel": kernel_name,
                "batch_size": 2,
                "seq_len": 16,
                "dtype": "fp16" if (self.device.type == "cuda" and torch.cuda.is_available()) else "fp32",
                "latency_ms": dispatched_ms,
                "latency_p50_ms": dispatched_ms,
                "latency_p90_ms": dispatched_ms,
                "latency_p99_ms": dispatched_ms,
                "tflops": 0.0,
                "peak_vram_mb": 0.0,
                "ref_ms": ref_ms,
                "speedup": speedup,
                "max_diff": max_diff,
                "status": status,
            }
            self.results_table.append(result_entry)
            return {"triton_ms": dispatched_ms, "ref_ms": ref_ms, "speedup": speedup}

        except Exception as e:
            logger.warning("Execution error in triton op %s: %s", kernel_name, e)
            self.results_table.append(
                {
                    "step": step,
                    "mechanism_or_kernel": kernel_name,
                    "batch_size": 2,
                    "seq_len": 16,
                    "dtype": "fp32",
                    "latency_ms": 0.0,
                    "latency_p50_ms": 0.0,
                    "latency_p90_ms": 0.0,
                    "latency_p99_ms": 0.0,
                    "tflops": 0.0,
                    "peak_vram_mb": 0.0,
                    "ref_ms": 0.0,
                    "speedup": 0.0,
                    "max_diff": 0.0,
                    "status": f"FAILED_RUNTIME: {e}",
                }
            )
            self.tracker.log_metric("triton_ms", 0.0, step=step)
            return {"triton_ms": 0.0, "ref_ms": 0.0, "speedup": 0.0}

    def run(self) -> None:
        step = 0
        for kernel in self.target_kernels:
            self.run_iteration(step=step, kernel_name=kernel)
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
        sorted_lat = sorted(passed_latencies)
        n = len(sorted_lat)
        p50 = sorted_lat[int(n * 0.50)] if n else 0.0
        p90 = sorted_lat[min(int(n * 0.90), n - 1)] if n else 0.0
        p99 = sorted_lat[min(int(n * 0.99), n - 1)] if n else 0.0

        status = "COMPLETED" if (passed > 0 or total == 0) else "FAILED"

        return ExecutionSummary(
            run_id=self.context.run_id,
            benchmark_name="triton_ops",
            status=status,
            total_iterations=total,
            passed_iterations=passed,
            failed_oom_iterations=0,
            failed_runtime_iterations=failed_runtime,
            mean_latency_ms=_sanitize_float(mean_lat),
            p50_latency_ms=_sanitize_float(p50),
            p90_latency_ms=_sanitize_float(p90),
            p99_latency_ms=_sanitize_float(p99),
            peak_vram_mb=0.0,
            mean_tflops=0.0,
            duration_seconds=duration,
            results_table=self.results_table,
            extra_metadata={"kernels_count": len(self.target_kernels)},
        )

    def teardown(self) -> None:
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
