#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Multi-Seed Statistical Significance & Cold-Cache Microbenchmarking (AGT-60).

Implements rigorous multi-seed randomization, cold-cache L2 eviction microbenchmarking,
dual-tier seed presets (Standard Canonical vs. Adversarial Stress Seeds), Student's t
and non-parametric bootstrap confidence intervals, exact Welch's two-sample t-tests
with Welch-Satterthwaite degrees of freedom, standardized Cohen's d effect sizes,
and LaTeX standard deviation reporting formatted according to academic publication rigor.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from neuralops.benchmarks.adapters.base import BaseBenchmarkAdapter
from neuralops.benchmarks.matrix import ColdCacheFlusher

logger = logging.getLogger(__name__)

# --- Seed Taxonomy & Presets ---
TIER1_CANONICAL_SEEDS: tuple[int, ...] = (42, 123, 456)
TIER2_ADVERSARIAL_SEEDS: tuple[int, ...] = (404, 777, 999)
DUAL_TIER_SEEDS: tuple[int, ...] = TIER1_CANONICAL_SEEDS + TIER2_ADVERSARIAL_SEEDS
ENTERPRISE_5_SEEDS: tuple[int, ...] = (42, 1337, 2026, 31415, 99999)
CANONICAL_7_SEEDS: tuple[int, ...] = (42, 100, 200, 300, 404, 777, 999)

SEED_PRESETS: dict[str, tuple[int, ...]] = {
    "canonical3": TIER1_CANONICAL_SEEDS,
    "adversarial3": TIER2_ADVERSARIAL_SEEDS,
    "dual_tier": DUAL_TIER_SEEDS,
    "enterprise5": ENTERPRISE_5_SEEDS,
    "canonical7": CANONICAL_7_SEEDS,
}


def set_deterministic_seed(seed: int) -> None:
    """Sets deterministic execution state across Python, random, NumPy, and PyTorch (CPU & CUDA)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def format_latex_sd(mean: float, std: float, unit: str = "%", precision: int = 2) -> str:
    """Format sample mean and sample standard deviation according to NeuralOps LaTeX rigor rules.

    Mandatory syntax:
      - unit == "%" -> "$86.40 \\pm 0.32\\%$"
      - unit in ("x", "\\times") -> "$1.24 \\pm 0.03\\times$"
      - unit == "ms" -> "$4.12 \\pm 0.04\\text{ ms}$"
      - unit == "tokens/s" -> "$12543.20 \\pm 112.50\\text{ tokens/s}$"
      - unit == "MB" -> "$1024.50 \\pm 4.20\\text{ MB}$"
      - unit == "" -> "$27.33 \\pm 0.07$"
    """
    mean_str = f"{mean:.{precision}f}"
    std_str = f"{std:.{precision}f}"

    if unit == "%":
        return f"${mean_str} \\pm {std_str}\\%$"
    elif unit in ("x", "\\times"):
        return f"${mean_str} \\pm {std_str}\\times$"
    elif unit:
        return f"${mean_str} \\pm {std_str}\\text{{ {unit}}}$"
    else:
        return f"${mean_str} \\pm {std_str}$"


def compute_students_t_ci(
    samples: Sequence[float], alpha: float = 0.05
) -> tuple[float, float]:
    """Computes exact two-sided Student's t confidence interval for the sample mean."""
    arr = np.asarray(samples, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return (0.0, 0.0)
    if n == 1 or np.var(arr, ddof=1) == 0.0:
        val = float(arr[0])
        return (val, val)

    from scipy import stats

    mean = float(np.mean(arr))
    se = float(stats.sem(arr))
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=n - 1))
    lower = mean - t_crit * se
    upper = mean + t_crit * se
    return (float(lower), float(upper))


def compute_bootstrap_ci(
    samples: Sequence[float],
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Computes non-parametric percentile bootstrap confidence interval for the sample mean."""
    arr = np.asarray(samples, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return (0.0, 0.0)
    if n == 1 or np.var(arr) == 0.0:
        val = float(arr[0])
        return (val, val)

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(n_boot, n))
    boot_means = np.mean(arr[indices], axis=1)

    alpha = 1.0 - ci
    lower = float(np.percentile(boot_means, 100 * (alpha / 2.0)))
    upper = float(np.percentile(boot_means, 100 * (1.0 - alpha / 2.0)))
    return (lower, upper)


def compute_cohens_d(sample_a: Sequence[float], sample_b: Sequence[float]) -> float:
    """Computes standardized Cohen's d effect size between sample A and sample B."""
    a = np.asarray(sample_a, dtype=np.float64)
    b = np.asarray(sample_b, dtype=np.float64)
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return 0.0
    m1, m2 = float(np.mean(a)), float(np.mean(b))
    v1, v2 = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
    s_pooled = math.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    return float((m1 - m2) / s_pooled) if s_pooled > 0 else 0.0


class WelchTestResult(dict):
    """Dictionary representing Welch's test results that allows both dict and attribute access."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'WelchTestResult' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def compute_welch_ttest(
    sample_a: Sequence[float],
    sample_b: Sequence[float],
    alpha: float = 0.05,
) -> WelchTestResult:
    """Computes two-sample Welch's t-test with exact Welch-Satterthwaite degrees of freedom."""
    try:
        from neuralops.analysis.comparison.statistical_testing import (
            welch_two_sample_ttest,
        )

        res = welch_two_sample_ttest(sample_a, sample_b, alpha=alpha)
        return WelchTestResult(res)
    except Exception:
        # Fallback to local exact implementation
        a = np.asarray(sample_a, dtype=np.float64)
        b = np.asarray(sample_b, dtype=np.float64)
        n1, n2 = len(a), len(b)
        if n1 < 2 or n2 < 2:
            return WelchTestResult({
                "method": "Welch's two-sample t-test",
                "difference": 0.0,
                "t_statistic": 0.0,
                "degrees_of_freedom": float(n1 + n2 - 2),
                "p_value": 1.0,
                "significant": False,
                "ci_95": (0.0, 0.0),
                "cohens_d": 0.0,
            })
        from scipy import stats

        m1, m2 = float(np.mean(a)), float(np.mean(b))
        v1, v2 = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
        diff = m1 - m2
        se_diff = math.sqrt(v1 / n1 + v2 / n2)
        num = (v1 / n1 + v2 / n2) ** 2
        den = ((v1 / n1) ** 2) / (n1 - 1) + ((v2 / n2) ** 2) / (n2 - 1)
        df = num / den if den > 0 else float(n1 + n2 - 2)
        t_stat = diff / se_diff if se_diff > 0 else 0.0
        p_val = 2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=df))
        t_crit = stats.t.ppf(1.0 - alpha / 2.0, df=df)
        ci_lower = diff - t_crit * se_diff
        ci_upper = diff + t_crit * se_diff
        s_pooled = math.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
        cohen_d = diff / s_pooled if s_pooled > 0 else 0.0
        return WelchTestResult({
            "method": "Welch's two-sample t-test",
            "mean_a": m1,
            "mean_b": m2,
            "difference": diff,
            "t_statistic": float(t_stat),
            "degrees_of_freedom": float(df),
            "p_value": float(p_val),
            "significant": bool(p_val < alpha),
            "ci_95": (float(ci_lower), float(ci_upper)),
            "cohens_d": float(cohen_d),
            "n_a": n1,
            "n_b": n2,
        })


@dataclass
class MultiSeedConfig:
    """Configuration for multi-seed statistical significance benchmark."""

    modules: list[str] = field(default_factory=lambda: ["attention:full_attention"])
    seeds: list[int] = field(default_factory=lambda: list(TIER1_CANONICAL_SEEDS))
    preset: Optional[str] = None
    baseline_module: Optional[str] = None
    batch_size: int = 1
    seq_len: int = 512
    hidden_dim: int = 64
    dtype: str = "float32"
    causal: bool = False
    device: str = "cpu"
    warmup: int = 5
    iters: int = 20
    flush_l2: bool = False
    l2_cache_size_mb: int = 256
    alpha: float = 0.05
    n_boot: int = 1000
    output_dir: str = "results/multiseed"
    tolerance: float = 1e-4
    dry_run: bool = False

    def __post_init__(self) -> None:
        if self.preset and self.preset in SEED_PRESETS:
            self.seeds = list(SEED_PRESETS[self.preset])


@dataclass
class SeedRunResult:
    """Execution metrics from a single seed evaluation run."""

    module: str
    seed: int
    status: str
    latency_ms: float = 0.0
    throughput_tokens_per_s: float = 0.0
    peak_memory_mb: float = 0.0
    error: Optional[str] = None
    raw_metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricAggregate:
    """Statistical summary across seeds for a specific metric."""

    metric_name: str
    unit: str
    values: list[float]
    mean: float
    std: float
    variance: float
    ci_95_student_t: tuple[float, float]
    ci_95_bootstrap: tuple[float, float]
    latex_formatted: str


@dataclass
class MultiSeedModuleResult:
    """Aggregated statistical evaluation of a single module across all seeds."""

    module: str
    seeds: list[int]
    status: str
    metrics: dict[str, MetricAggregate]
    baseline_comparison: Optional[dict[str, Any]] = None
    seed_measurements: list[SeedRunResult] = field(default_factory=list)


@dataclass
class MultiSeedBenchmarkResult:
    """Complete multi-seed statistical benchmark report across all modules."""

    config: MultiSeedConfig
    module_results: dict[str, MultiSeedModuleResult]
    raw_runs: list[SeedRunResult]
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Convert entire multi-seed result to serializable dictionary."""
        return {
            "config": asdict(self.config),
            "timestamp": self.timestamp,
            "module_results": {
                mod: {
                    "module": res.module,
                    "seeds": res.seeds,
                    "status": res.status,
                    "metrics": {
                        mname: asdict(magg) for mname, magg in res.metrics.items()
                    },
                    "baseline_comparison": res.baseline_comparison,
                    "seed_measurements": [asdict(m) for m in res.seed_measurements],
                }
                for mod, res in self.module_results.items()
            },
            "raw_runs": [asdict(r) for r in self.raw_runs],
        }

    def to_latex_table(self) -> str:
        """Generates a camera-ready publication LaTeX table comparing module results with standard deviations."""
        lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            r"\caption{Multi-Seed Statistical Significance Evaluation across "
            + f"{len(self.config.seeds)}"
            + r" independent seeds.}\label{tab:multiseed_rigor}",
            r"\begin{tabular}{lcccc}",
            r"\toprule",
            r"\textbf{Module} & \textbf{Latency (ms)} & \textbf{Throughput (tokens/s)} & \textbf{Welch's $p$-value} & \textbf{Cohen's $d$} \\",
            r"\midrule",
        ]

        for mod, res in self.module_results.items():
            lat_agg = res.metrics.get("latency_ms")
            tput_agg = res.metrics.get("throughput_tokens_per_s")

            lat_str = lat_agg.latex_formatted if lat_agg else "N/A"
            tput_str = tput_agg.latex_formatted if tput_agg else "N/A"

            p_val_str = "Baseline"
            cohen_str = "---"
            if res.baseline_comparison:
                p_val = res.baseline_comparison.get("p_value", 1.0)
                d = res.baseline_comparison.get("cohens_d", 0.0)
                p_val_str = (
                    f"$< 10^{{-4}}$" if p_val < 1e-4 else f"${p_val:.4f}$"
                )
                if res.baseline_comparison.get("significant", False):
                    p_val_str += r"^\ast"
                cohen_str = f"${d:+.2f}$"

            clean_mod = mod
            lines.append(
                f"{clean_mod} & {lat_str} & {tput_str} & {p_val_str} & {cohen_str} \\\\"
            )

        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])
        return "\n".join(lines)

    def save(self, path: Optional[Union[str, Path]] = None) -> Path:
        """Persist results to JSON and LaTeX outputs."""
        if path is None:
            out_dir = Path(self.config.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "multiseed_results.json"
        else:
            out_path = Path(path)
            out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

        latex_path = out_path.with_suffix(".tex")
        with open(latex_path, "w", encoding="utf-8") as f:
            f.write(self.to_latex_table() + "\n")

        return out_path


class MultiSeedBenchmarkOrchestrator:
    """Orchestrates multi-seed executions, statistical significance testing, and cold-cache microbenchmarking."""

    def __init__(self, config: MultiSeedConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.flusher = (
            ColdCacheFlusher(self.device, size_mb=config.l2_cache_size_mb)
            if config.flush_l2
            else None
        )

    def _resolve_adapter(self, module_name: str) -> BaseBenchmarkAdapter:
        """Resolve and instantiate the appropriate 6-gate benchmark adapter."""
        from neuralops.benchmarks.adapters.attention import AttentionBenchmarkAdapter
        from neuralops.benchmarks.adapters.compressor import CompressorModuleAdapter
        from neuralops.benchmarks.adapters.cuda_adapter import (
            CANONICAL_CUDA_MODULES,
            CUDAKernelAdapter as CUDABenchmarkAdapter,
        )
        from neuralops.benchmarks.adapters.triton_adapter import (
            CANONICAL_TRITON_MODULES,
            TritonKernelAdapter as TritonBenchmarkAdapter,
        )
        from neuralops.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY

        mod = module_name
        dtype_torch = getattr(torch, self.config.dtype, torch.float32)

        if mod.startswith("attention:") or mod in ATTENTION_REGISTRY:
            clean_name = mod.split(":", 1)[1] if ":" in mod else mod
            return AttentionBenchmarkAdapter(
                attention_name=clean_name,
                device=self.device,
                dtype=dtype_torch,
                batch_size=self.config.batch_size,
                seq_len=self.config.seq_len,
                dim=self.config.hidden_dim,
                causal=self.config.causal,
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
                batch_size=self.config.batch_size,
                seq_len=self.config.seq_len,
                dim=self.config.hidden_dim,
                causal=self.config.causal,
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
                batch_size=self.config.batch_size,
                seq_len=self.config.seq_len,
                dim=self.config.hidden_dim,
                causal=self.config.causal,
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
                batch_size=self.config.batch_size,
                seq_len=self.config.seq_len,
                dim=self.config.hidden_dim,
                tolerance=self.config.tolerance,
            )
        else:
            clean_name = mod.split(":", 1)[1] if ":" in mod else mod
            return AttentionBenchmarkAdapter(
                attention_name=clean_name,
                device=self.device,
                dtype=dtype_torch,
                batch_size=self.config.batch_size,
                seq_len=self.config.seq_len,
                dim=self.config.hidden_dim,
                causal=self.config.causal,
                warmup_steps=self.config.warmup,
                measure_steps=self.config.iters,
                tolerance=self.config.tolerance,
            )

    def execute_seed_run(self, module_name: str, seed: int) -> SeedRunResult:
        """Execute a single seed benchmark with deterministic conditioning and cold cache eviction."""
        if self.config.dry_run:
            return SeedRunResult(
                module=module_name,
                seed=seed,
                status="PLANNED",
                latency_ms=1.0,
                throughput_tokens_per_s=1000.0,
                peak_memory_mb=10.0,
            )

        set_deterministic_seed(seed)

        if self.flusher:
            self.flusher.flush()

        try:
            adapter = self._resolve_adapter(module_name)
            res = adapter.execute()
            status_val = res.status.value if hasattr(res.status, "value") else str(res.status)
            if status_val.lower() != "success":
                return SeedRunResult(
                    module=module_name,
                    seed=seed,
                    status=status_val,
                    error=res.error_message,
                    raw_metrics=res.metrics,
                )

            latency_ms = float(res.metrics.get("mean_latency_ms", res.metrics.get("latency_ms", 0.0)))
            if latency_ms == 0.0 and "p50_latency_ms" in res.metrics:
                latency_ms = float(res.metrics["p50_latency_ms"])

            tokens = self.config.batch_size * self.config.seq_len
            throughput = (tokens / (latency_ms / 1000.0)) if latency_ms > 0 else 0.0
            peak_memory = float(res.metrics.get("peak_memory_mb", res.metrics.get("vram_allocated_mb", 0.0)))

            return SeedRunResult(
                module=module_name,
                seed=seed,
                status="SUCCESS",
                latency_ms=latency_ms,
                throughput_tokens_per_s=throughput,
                peak_memory_mb=peak_memory,
                raw_metrics=res.metrics,
            )
        except Exception as e:
            logger.error("Seed execution failure for %s seed %d: %s", module_name, seed, e)
            return SeedRunResult(
                module=module_name,
                seed=seed,
                status="ERROR",
                error=str(e),
            )

    def _aggregate_metric(
        self, metric_name: str, unit: str, values: list[float]
    ) -> MetricAggregate:
        """Compute statistical properties across seeds for a single metric."""
        arr = np.asarray(values, dtype=np.float64)
        n = len(arr)
        if n == 0:
            mean = std = var = 0.0
            ci_t = ci_boot = (0.0, 0.0)
        elif n == 1:
            mean = float(arr[0])
            std = var = 0.0
            ci_t = ci_boot = (mean, mean)
        else:
            mean = float(np.mean(arr))
            std = float(np.std(arr, ddof=1))
            var = float(np.var(arr, ddof=1))
            ci_t = compute_students_t_ci(arr, alpha=self.config.alpha)
            ci_boot = compute_bootstrap_ci(arr, n_boot=self.config.n_boot, ci=1.0 - self.config.alpha)

        latex_str = format_latex_sd(mean, std, unit=unit)
        return MetricAggregate(
            metric_name=metric_name,
            unit=unit,
            values=[float(x) for x in arr],
            mean=mean,
            std=std,
            variance=var,
            ci_95_student_t=ci_t,
            ci_95_bootstrap=ci_boot,
            latex_formatted=latex_str,
        )

    def run(self) -> MultiSeedBenchmarkResult:
        """Executes full multi-seed benchmark matrix across all modules and seeds."""
        raw_runs: list[SeedRunResult] = []
        module_results: dict[str, MultiSeedModuleResult] = {}

        for mod in self.config.modules:
            mod_runs: list[SeedRunResult] = []
            latencies: list[float] = []
            throughputs: list[float] = []
            memories: list[float] = []

            for s in self.config.seeds:
                run_res = self.execute_seed_run(mod, s)
                mod_runs.append(run_res)
                raw_runs.append(run_res)

                if run_res.status in ("SUCCESS", "PLANNED"):
                    latencies.append(run_res.latency_ms)
                    throughputs.append(run_res.throughput_tokens_per_s)
                    memories.append(run_res.peak_memory_mb)

            status = "PLANNED" if self.config.dry_run else ("SUCCESS" if latencies else "FAILED")
            metrics = {
                "latency_ms": self._aggregate_metric("latency_ms", "ms", latencies),
                "throughput_tokens_per_s": self._aggregate_metric("throughput_tokens_per_s", "tokens/s", throughputs),
                "peak_memory_mb": self._aggregate_metric("peak_memory_mb", "MB", memories),
            }

            module_results[mod] = MultiSeedModuleResult(
                module=mod,
                seeds=self.config.seeds,
                status=status,
                metrics=metrics,
                seed_measurements=mod_runs,
            )

        # Comparative analysis against baseline module if specified
        base_mod = self.config.baseline_module
        if base_mod and base_mod in module_results:
            base_lats = module_results[base_mod].metrics["latency_ms"].values
            if len(base_lats) >= 2:
                for mod, res in module_results.items():
                    if mod == base_mod:
                        continue
                    mod_lats = res.metrics["latency_ms"].values
                    if len(mod_lats) >= 2:
                        res.baseline_comparison = compute_welch_ttest(
                            mod_lats, base_lats, alpha=self.config.alpha
                        )

        result = MultiSeedBenchmarkResult(
            config=self.config,
            module_results=module_results,
            raw_runs=raw_runs,
        )

        if not self.config.dry_run and self.config.output_dir:
            result.save()

        return result


def run_multiseed_benchmark(
    modules: list[str],
    seeds: Optional[list[int]] = None,
    preset: Optional[str] = None,
    baseline_module: Optional[str] = None,
    batch_size: int = 1,
    seq_len: int = 512,
    hidden_dim: int = 64,
    dtype: str = "float32",
    causal: bool = False,
    device: str = "cpu",
    warmup: int = 5,
    iters: int = 20,
    flush_l2: bool = False,
    l2_cache_size_mb: int = 256,
    alpha: float = 0.05,
    n_boot: int = 1000,
    output_dir: str = "results/multiseed",
    dry_run: bool = False,
) -> MultiSeedBenchmarkResult:
    """Convenience functional entry point for multi-seed statistical benchmarks."""
    actual_seeds = list(seeds) if seeds else (list(SEED_PRESETS.get(preset, TIER1_CANONICAL_SEEDS)) if preset else list(TIER1_CANONICAL_SEEDS))
    config = MultiSeedConfig(
        modules=modules,
        seeds=actual_seeds,
        preset=preset,
        baseline_module=baseline_module,
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        dtype=dtype,
        causal=causal,
        device=device,
        warmup=warmup,
        iters=iters,
        flush_l2=flush_l2,
        l2_cache_size_mb=l2_cache_size_mb,
        alpha=alpha,
        n_boot=n_boot,
        output_dir=output_dir,
        dry_run=dry_run,
    )
    orchestrator = MultiSeedBenchmarkOrchestrator(config)
    return orchestrator.run()
