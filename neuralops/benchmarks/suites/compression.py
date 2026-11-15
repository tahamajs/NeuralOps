#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Compression benchmark suite sweeping registered model compression engines.
"""

from __future__ import annotations

import copy
import logging
import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from neuralops.benchmarks.core.base import AbstractBaseBenchmark, ExecutionSummary, _sanitize_float
from neuralops.config.benchmark_schema import CompressionSweepParameters

try:
    from neuralops.compressors import COMPRESSOR_REGISTRY
except ImportError:
    try:
        from neuralops.compression import COMPRESSOR_REGISTRY
    except ImportError:
        from neuralops.core.registry import COMPRESSOR_REGISTRY

from neuralops.core.registry import create_compressor

logger = logging.getLogger(__name__)


class _BenchmarkHostModule(nn.Module):
    def __init__(self, in_features: int = 64, out_features: int = 64) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(out_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


def _count_module_params(model: nn.Module, compressor: Any = None) -> int:
    """Count model parameters, ensuring factorized SVD components (U, V) are accurately counted."""
    counted_ids: set[int] = set()
    total = 0

    if compressor is not None:
        for attr in ("U", "V"):
            t = getattr(compressor, attr, None)
            if isinstance(t, torch.Tensor) and id(t) not in counted_ids:
                counted_ids.add(id(t))
                total += t.numel()

    for m in model.modules():
        for p in m.parameters(recurse=False):
            if id(p) not in counted_ids:
                counted_ids.add(id(p))
                total += p.numel()
        for attr in ("U", "V"):
            t = getattr(m, attr, None)
            if isinstance(t, torch.Tensor) and id(t) not in counted_ids:
                counted_ids.add(id(t))
                total += t.numel()

    return total


class CompressionBenchmark(AbstractBaseBenchmark):
    def setup(self) -> None:
        sweep_cfg = self.config.compression_sweep or CompressionSweepParameters()
        custom = self.config.custom_parameters or {}

        cfg_comps = custom.get("compressors", sweep_cfg.compressors)
        if cfg_comps == ["all"] or not cfg_comps:
            self.compressors = sorted(list(COMPRESSOR_REGISTRY.keys()))
        else:
            self.compressors = list(cfg_comps)

        limit = custom.get("limit")
        if limit is not None and isinstance(limit, int) and limit > 0:
            self.compressors = self.compressors[:limit]

        self.target_ratios = custom.get("target_ratios", sweep_cfg.target_ratios)
        self.dim = custom.get("hidden_size", 64)
        self.warmup_iterations = custom.get("warmup_iterations", 2)
        self.benchmark_iterations = custom.get("benchmark_iterations", 5)

        self.start_time = time.time()
        self.results_table: List[Dict[str, Any]] = []

    def run_iteration(
        self,
        step: int,
        compressor_name: str,
        target_ratio: float = 0.50,
        **kwargs: Any,
    ) -> Dict[str, float]:
        self.flush_l2_cache()

        torch.manual_seed(42 + step)
        host_model = _BenchmarkHostModule(self.dim, self.dim).to(device=self.device)
        total_orig_params = _count_module_params(host_model)

        rank = max(4, int(self.dim * (1.0 - target_ratio)))

        try:
            t0 = time.perf_counter()
            compressor = create_compressor(
                compressor_name,
                model=host_model,
                rank=rank,
                target_ratio=target_ratio,
            )

            if hasattr(compressor, "compress"):
                comp_out = compressor.compress()
                if isinstance(comp_out, nn.Module):
                    host_model = comp_out
            elif hasattr(compressor, "apply"):
                comp_out = compressor.apply(host_model)
                if isinstance(comp_out, nn.Module):
                    host_model = comp_out
            t1 = time.perf_counter()

            compression_latency_ms = _sanitize_float((t1 - t0) * 1000.0)

            x = torch.randn(2, 16, self.dim, device=self.device)
            with torch.no_grad():
                out = host_model(x)
                if not torch.isfinite(out).all():
                    raise ValueError(f"Non-finite output from compressed model using {compressor_name}")

            def forward_call() -> None:
                with torch.no_grad():
                    host_model(x)

            forward_latency_ms, _ = self.measure_latency(
                forward_call,
                warmup=self.warmup_iterations,
                repeat=self.benchmark_iterations,
            )

            total_compressed_params = _count_module_params(host_model, compressor)
            param_reduction = max(0.0, 1.0 - (total_compressed_params / max(total_orig_params, 1)))

            self.tracker.log_metric("compression_latency_ms", compression_latency_ms, step=step)

            result_entry = {
                "step": step,
                "mechanism_or_kernel": compressor_name,
                "batch_size": 2,
                "seq_len": 16,
                "dtype": "fp32",
                "latency_ms": compression_latency_ms,
                "latency_p50_ms": compression_latency_ms,
                "latency_p90_ms": compression_latency_ms,
                "latency_p99_ms": compression_latency_ms,
                "tflops": 0.0,
                "peak_vram_mb": 0.0,
                "forward_latency_ms": forward_latency_ms,
                "param_reduction_pct": round(param_reduction * 100.0, 2),
                "status": "PASS",
            }
            self.results_table.append(result_entry)
            return {
                "compression_latency_ms": compression_latency_ms,
                "forward_latency_ms": forward_latency_ms,
            }

        except Exception as e:
            logger.warning("Compression failed for '%s': %s", compressor_name, e)
            self.results_table.append(
                {
                    "step": step,
                    "mechanism_or_kernel": compressor_name,
                    "batch_size": 2,
                    "seq_len": 16,
                    "dtype": "fp32",
                    "latency_ms": 0.0,
                    "latency_p50_ms": 0.0,
                    "latency_p90_ms": 0.0,
                    "latency_p99_ms": 0.0,
                    "tflops": 0.0,
                    "peak_vram_mb": 0.0,
                    "status": f"FAILED_RUNTIME: {e}",
                }
            )
            self.tracker.log_metric("compression_latency_ms", 0.0, step=step)
            return {"compression_latency_ms": 0.0, "forward_latency_ms": 0.0}

    def run(self) -> None:
        step = 0
        ratio = self.target_ratios[0] if self.target_ratios else 0.50
        for comp in self.compressors:
            self.run_iteration(step=step, compressor_name=comp, target_ratio=ratio)
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
            benchmark_name="compression",
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
            extra_metadata={"compressors_count": len(self.compressors)},
        )

    def teardown(self) -> None:
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
