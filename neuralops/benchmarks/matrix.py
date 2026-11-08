#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Multi-Phase Benchmark Matrix Orchestrator.

Orchestrates multi-dimensional benchmarking grids across shapes (batch size B,
sequence length T, hidden dimension D), precisions (FP16, BF16, FP32), and causal
masking modes. Implements GPU synchronous event timing, L2 cache eviction protocol,
statistical percentile aggregation, derived throughput/FLOPs metrics, and prefill
speedup regression gating.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from neuralops.benchmarks.adapters.attention import AttentionBenchmarkAdapter
from neuralops.benchmarks.adapters.base import BaseBenchmarkAdapter
from neuralops.benchmarks.adapters.compressor import CompressorModuleAdapter
from neuralops.benchmarks.adapters.cuda_adapter import CANONICAL_CUDA_MODULES, CUDAKernelAdapter as CUDABenchmarkAdapter
from neuralops.benchmarks.adapters.triton_adapter import CANONICAL_TRITON_MODULES, TritonKernelAdapter as TritonBenchmarkAdapter
from neuralops.benchmarks.results.schema import BenchmarkResult, BenchmarkStatus
from neuralops.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY

logger = logging.getLogger(__name__)


@dataclass
class MatrixDimensionConfig:
    """Configures the parameter axes of the benchmark matrix."""

    seq_lens: List[int] = field(default_factory=lambda: [128, 512, 1024, 2048])
    batch_sizes: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    hidden_dims: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    dtypes: List[str] = field(default_factory=lambda: ["float32"])
    causal_modes: List[bool] = field(default_factory=lambda: [True, False])
    modules: List[str] = field(default_factory=list)
    warmup: int = 5
    iters: int = 20
    flush_l2: bool = True
    l2_cache_size_mb: int = 256
    device: str = "cpu"
    tolerance: float = 1e-4
    output_dir: str = "results/matrix"


@dataclass(frozen=True)
class MatrixPoint:
    """A single evaluation cell within the multi-dimensional benchmark grid."""

    module: str
    seq_len: int
    batch_size: int
    hidden_dim: int
    dtype: str
    causal: bool

    @property
    def regime(self) -> str:
        """Classify workload regime: prefill vs decode vs micro."""
        if self.seq_len >= 512:
            return "prefill"
        elif self.batch_size == 1 and self.seq_len >= 1024:
            return "decode"
        return "micro"


@dataclass
class MatrixCellResult:
    """Comprehensive performance telemetry and verification for a single grid point."""

    point: MatrixPoint
    status: str
    regime: str
    latency_ms_median: float = 0.0
    latency_ms_p10: float = 0.0
    latency_ms_p90: float = 0.0
    latency_ms_mean: float = 0.0
    latency_ms_std: float = 0.0
    throughput_tokens_per_s: float = 0.0
    tflops: float = 0.0
    effective_bandwidth_gb_s: float = 0.0
    speedup_vs_reference: float = 1.0
    peak_memory_mb: float = 0.0
    numerics_passed: bool = True
    speedup_gate_passed: bool = True
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["point"] = asdict(self.point)
        return d


class ColdCacheFlusher:
    """Manages L2 cache eviction protocol by allocating a buffer larger than hardware L2 capacity."""

    def __init__(self, device: torch.device, size_mb: int = 256) -> None:
        self.device = device
        self.size_mb = size_mb
        self._buffer: Optional[torch.Tensor] = None
        if device.type == "cuda" and torch.cuda.is_available():
            try:
                num_elements = (size_mb * 1024 * 1024) // 4
                self._buffer = torch.empty(num_elements, dtype=torch.int32, device=device)
            except Exception as e:
                logger.debug("ColdCacheFlusher buffer allocation skipped: %s", e)
                self._buffer = None

    def flush(self) -> None:
        """Evict device cache lines."""
        if self._buffer is not None:
            self._buffer.zero_()


class BenchmarkMatrixOrchestrator:
    """5-phase multi-axis benchmark grid orchestrator.

    Executes:
      Phase 1: Grid Dimension Generation & Workload Regimes
      Phase 2: Cold-Cache Flushing & Hardware Timing (CUDA Events / Perf Counter)
      Phase 3: Execution Across Adapter Fallback Hierarchy
      Phase 4: Metric Derivation (Tokens/s, TFLOPs, Memory BW, Speedup)
      Phase 5: Performance Gating & Summary Reporting
    """

    def __init__(self, config: MatrixDimensionConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.flusher = ColdCacheFlusher(self.device, size_mb=config.l2_cache_size_mb) if config.flush_l2 else None
        self._results: List[MatrixCellResult] = []

    def plan(self) -> List[MatrixPoint]:
        """Generate grid points without executing (dry-run)."""
        modules = self.config.modules
        if not modules:
            # Default to baseline attention
            modules = ["attention:full_attention"]

        points: List[MatrixPoint] = []
        for mod in modules:
            for s in self.config.seq_lens:
                for b in self.config.batch_sizes:
                    for d in self.config.hidden_dims:
                        for dt in self.config.dtypes:
                            for c in self.config.causal_modes:
                                points.append(
                                    MatrixPoint(
                                        module=mod,
                                        seq_len=s,
                                        batch_size=b,
                                        hidden_dim=d,
                                        dtype=dt,
                                        causal=c,
                                    )
                                )
        return points

    def _resolve_adapter(self, point: MatrixPoint) -> BaseBenchmarkAdapter:
        """Resolve and instantiate the appropriate 6-gate benchmark adapter."""
        mod = point.module
        dtype_torch = getattr(torch, point.dtype, torch.float32)

        if mod.startswith("attention:") or mod in ATTENTION_REGISTRY:
            clean_name = mod.split(":", 1)[1] if ":" in mod else mod
            return AttentionBenchmarkAdapter(
                attention_name=clean_name,
                device=self.device,
                dtype=dtype_torch,
                batch_size=point.batch_size,
                seq_len=point.seq_len,
                dim=point.hidden_dim,
                causal=point.causal,
                warmup_steps=self.config.warmup,
                measure_steps=self.config.iters,
                tolerance=self.config.tolerance,
            )
        elif mod.startswith("triton:") or mod in CANONICAL_TRITON_MODULES:
            clean_name = mod.split(":", 1)[1] if ":" in mod else mod
            return TritonBenchmarkAdapter(
                module_name=clean_name,
                device=self.device,
                dtype=dtype_torch,
                batch_size=point.batch_size,
                seq_len=point.seq_len,
                dim=point.hidden_dim,
                causal=point.causal,
                warmup_steps=self.config.warmup,
                measure_steps=self.config.iters,
                tolerance=self.config.tolerance,
            )
        elif mod.startswith("cuda:") or mod in CANONICAL_CUDA_MODULES:
            clean_name = mod.split(":", 1)[1] if ":" in mod else mod
            return CUDABenchmarkAdapter(
                module_name=clean_name,
                device=self.device,
                dtype=dtype_torch,
                batch_size=point.batch_size,
                seq_len=point.seq_len,
                dim=point.hidden_dim,
                causal=point.causal,
                warmup_steps=self.config.warmup,
                measure_steps=self.config.iters,
                tolerance=self.config.tolerance,
            )
        elif mod.startswith("compressor:") or mod in COMPRESSOR_REGISTRY:
            clean_name = mod.split(":", 1)[1] if ":" in mod else mod
            return CompressorModuleAdapter(
                compressor_name=clean_name,
                device=self.device,
                dtype=dtype_torch,
                batch_size=point.batch_size,
                seq_len=point.seq_len,
                dim=point.hidden_dim,
                tolerance=self.config.tolerance,
            )
        else:
            # Fallback to attention resolution
            clean_name = mod.split(":", 1)[1] if ":" in mod else mod
            return AttentionBenchmarkAdapter(
                attention_name=clean_name,
                device=self.device,
                dtype=dtype_torch,
                batch_size=point.batch_size,
                seq_len=point.seq_len,
                dim=point.hidden_dim,
                causal=point.causal,
                warmup_steps=self.config.warmup,
                measure_steps=self.config.iters,
                tolerance=self.config.tolerance,
            )

    def execute_cell(self, point: MatrixPoint) -> MatrixCellResult:
        """Execute a single point in the matrix and measure multi-phase performance metrics."""
        regime = point.regime
        try:
            adapter = self._resolve_adapter(point)

            # Optional cold-cache eviction
            if self.flusher:
                self.flusher.flush()

            # Execute 6-gate adapter lifecycle
            res = adapter.execute()
            status_str = str(res.status).lower()

            if status_str not in ("success", BenchmarkStatus.SUCCESS.value.lower()):
                return MatrixCellResult(
                    point=point,
                    status=status_str,
                    regime=regime,
                    numerics_passed=False,
                    speedup_gate_passed=False,
                    error=res.error or "Adapter lifecycle failed",
                )

            meta = res.metadata or {}
            times_ms = meta.get("latencies_ms", [])
            if not times_ms:
                mean_lat = meta.get("latency_ms_mean") or meta.get("forward_latency_ms") or 1.0
                times_ms = [mean_lat]

            times_sorted = sorted(times_ms)
            n = len(times_sorted)
            median_ms = meta.get("latency_ms_median", times_sorted[n // 2])
            p10_ms = times_sorted[max(0, int(n * 0.10))]
            p90_ms = times_sorted[min(n - 1, int(n * 0.90))]
            mean_ms = meta.get("latency_ms_mean", statistics.mean(times_ms))
            std_ms = meta.get("latency_ms_std", statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0)

            median_s = median_ms / 1000.0 if median_ms > 0 else 0.001

            # Derived Metrics (Phase 4):
            # 1. Throughput (Tokens/s): (B * T) / Latency(s)
            b_total_tokens = point.batch_size * point.seq_len
            throughput = meta.get(
                "throughput_tokens_per_sec",
                meta.get("throughput_tokens_per_s", b_total_tokens / median_s if median_s > 0 else 0.0),
            )

            # 2. Arithmetic Intensity / TFLOPs:
            # Standard attention FLOPs: 4 * B * T^2 * D + 2 * B * T * D
            total_flops = 4.0 * point.batch_size * (point.seq_len**2) * point.hidden_dim
            tflops = (total_flops / (median_s * 1e12)) if median_s > 0 else 0.0

            # 3. Effective Memory Bandwidth (GB/s):
            bytes_per_elem = 2 if "16" in point.dtype else 4
            bytes_transferred = 2.0 * b_total_tokens * point.hidden_dim * bytes_per_elem
            bw_gb_s = (bytes_transferred / (median_s * 1e9)) if median_s > 0 else 0.0

            # 4. Speedup Ratio vs PyTorch reference baseline:
            speedup = meta.get("speedup_vs_baseline", meta.get("speedup_vs_reference", 1.0))
            if speedup is None:
                speedup = 1.0

            # Phase 5: Gating & Anomaly Detection:
            speedup_gate = True
            if regime == "prefill" and point.seq_len >= 1024 and speedup < 0.90:
                speedup_gate = False
                logger.warning(
                    "Speedup regression detected at T=%d for %s: speedup=%.2fx < 0.90x",
                    point.seq_len,
                    point.module,
                    speedup,
                )

            peak_mem = meta.get("memory_mb_peak", meta.get("peak_memory_mb", 0.0))

            return MatrixCellResult(
                point=point,
                status="success",
                regime=regime,
                latency_ms_median=round(float(median_ms), 4),
                latency_ms_p10=round(float(p10_ms), 4),
                latency_ms_p90=round(float(p90_ms), 4),
                latency_ms_mean=round(float(mean_ms), 4),
                latency_ms_std=round(float(std_ms), 4),
                throughput_tokens_per_s=round(float(throughput), 2),
                tflops=round(float(tflops), 4),
                effective_bandwidth_gb_s=round(float(bw_gb_s), 4),
                speedup_vs_reference=round(float(speedup), 2),
                peak_memory_mb=round(float(peak_mem), 2),
                numerics_passed=True,
                speedup_gate_passed=speedup_gate,
            )

        except Exception as exc:
            logger.error("Execution failed for cell %s: %s", point, exc)
            return MatrixCellResult(
                point=point,
                status="failed",
                regime=regime,
                numerics_passed=False,
                speedup_gate_passed=False,
                error=str(exc),
            )

    def execute_grid(self) -> List[MatrixCellResult]:
        """Execute the entire multi-axis benchmark matrix grid."""
        points = self.plan()
        logger.info("Executing benchmark matrix grid of %d points...", len(points))
        self._results = []

        out_path = Path(self.config.output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        jsonl_file = out_path / "matrix_results.jsonl"

        for idx, pt in enumerate(points, 1):
            logger.info("[%d/%d] Testing %s (B=%d, T=%d, D=%d, %s, causal=%s)...",
                        idx, len(points), pt.module, pt.batch_size, pt.seq_len, pt.hidden_dim, pt.dtype, pt.causal)
            res = self.execute_cell(pt)
            self._results.append(res)

            # Incremental persistence
            with open(jsonl_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(res.to_dict()) + "\n")

        self.export_summary()
        return self._results

    def export_summary(self) -> Dict[str, Any]:
        """Generate tabular summary, CSV, and summary.json for the sweep."""
        out_path = Path(self.config.output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        summary_file = out_path / "summary.json"
        csv_file = out_path / "matrix_summary.csv"

        total = len(self._results)
        passed = sum(1 for r in self._results if r.status == "success" and r.numerics_passed)
        speedup_failures = sum(1 for r in self._results if not r.speedup_gate_passed)

        summary_data = {
            "total_cells": total,
            "passed_cells": passed,
            "failed_cells": total - passed,
            "speedup_gate_failures": speedup_failures,
            "device": str(self.device),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "regimes": {
                "prefill": sum(1 for r in self._results if r.regime == "prefill"),
                "decode": sum(1 for r in self._results if r.regime == "decode"),
                "micro": sum(1 for r in self._results if r.regime == "micro"),
            },
        }

        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)

        # Write CSV
        if self._results:
            fieldnames = [
                "module", "regime", "batch_size", "seq_len", "hidden_dim", "dtype", "causal",
                "status", "latency_ms_median", "throughput_tokens_per_s", "tflops",
                "effective_bandwidth_gb_s", "speedup_vs_reference", "speedup_gate_passed", "error"
            ]
            with open(csv_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in self._results:
                    writer.writerow({
                        "module": r.point.module,
                        "regime": r.regime,
                        "batch_size": r.point.batch_size,
                        "seq_len": r.point.seq_len,
                        "hidden_dim": r.point.hidden_dim,
                        "dtype": r.point.dtype,
                        "causal": r.point.causal,
                        "status": r.status,
                        "latency_ms_median": r.latency_ms_median,
                        "throughput_tokens_per_s": r.throughput_tokens_per_s,
                        "tflops": r.tflops,
                        "effective_bandwidth_gb_s": r.effective_bandwidth_gb_s,
                        "speedup_vs_reference": r.speedup_vs_reference,
                        "speedup_gate_passed": r.speedup_gate_passed,
                        "error": r.error or "",
                    })

        return summary_data


def run_benchmark_matrix(
    modules: Optional[List[str]] = None,
    seq_lens: Optional[List[int]] = None,
    batch_sizes: Optional[List[int]] = None,
    hidden_dims: Optional[List[int]] = None,
    dtypes: Optional[List[str]] = None,
    causal_modes: Optional[List[bool]] = None,
    device: str = "cpu",
    output_dir: str = "results/matrix",
    dry_run: bool = False,
    warmup: int = 5,
    iters: int = 20,
    flush_l2: bool = True,
) -> Dict[str, Any]:
    """Top-level procedural entrypoint for orchestrating a benchmark matrix."""
    config = MatrixDimensionConfig(
        seq_lens=seq_lens or [128, 512, 1024, 2048],
        batch_sizes=batch_sizes or [1, 2, 4, 8],
        hidden_dims=hidden_dims or [64, 128, 256, 512],
        dtypes=dtypes or ["float32"],
        causal_modes=causal_modes if causal_modes is not None else [True, False],
        modules=modules or ["attention:full_attention"],
        device=device,
        output_dir=output_dir,
        warmup=warmup,
        iters=iters,
        flush_l2=flush_l2,
    )
    orchestrator = BenchmarkMatrixOrchestrator(config)
    if dry_run:
        plan = orchestrator.plan()
        return {
            "dry_run": True,
            "total_points": len(plan),
            "points": [asdict(p) for p in plan],
        }

    results = orchestrator.execute_grid()
    return orchestrator.export_summary()
