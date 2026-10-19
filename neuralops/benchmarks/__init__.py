#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
NeuralOps Benchmarks Subpackage.
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "benchmark",
    "Runner",
    "BenchmarkResult",
    "BenchmarkConfig",
    "BenchmarkRunConfig",
    "QualitySweepConfig",
    "RankSweepConfig",
    "MultiModeConfig",
    "MultiseedConfig",
    "SeqLenScalingConfig",
    "RulerConfig",
    "GPT2SweepConfig",
    "KDTestConfig",
    "main",
    "LMEvalAdapter",
    "RULERBenchmark",
    "CustomEvaluationSuite",
    "run_gpt2_sweep",
    "run_quality_sweep",
    "BenchmarkRunner",
    "BenchmarkMetric",
    "HardwareProfile",
    "bench_llm_bandwidth",
    "benchmark_torch_compile",
    "benchmark_all_26_comprehensive",
    "benchmark_all_26_kernels",
    "benchmark_extrapolation_32k",
    "benchmark_dynamic_range_stress",
    "measure_multiprecision",
    "multi_seed_stress_test_131_72",
    "stress_test_seeds_100_200_300",
    "stress_test_seeds_404_777_999",
    "benchmark_torch_compile_131_attentions",
]

_LAZY_BENCHMARK_SYMBOLS: dict[str, tuple[str, str]] = {
    "Runner": ("neuralops.benchmarks.api", "Runner"),
    "benchmark": ("neuralops.benchmarks.api", "benchmark"),
    "main": ("neuralops.benchmarks.cli", "main"),
    "BenchmarkResult": ("neuralops.benchmarks.result", "BenchmarkResult"),
    "BenchmarkConfig": ("neuralops.benchmarks.config", "BenchmarkConfig"),
    "BenchmarkRunConfig": ("neuralops.benchmarks.config", "BenchmarkRunConfig"),
    "QualitySweepConfig": ("neuralops.benchmarks.config", "QualitySweepConfig"),
    "RankSweepConfig": ("neuralops.benchmarks.config", "RankSweepConfig"),
    "MultiModeConfig": ("neuralops.benchmarks.config", "MultiModeConfig"),
    "MultiseedConfig": ("neuralops.benchmarks.config", "MultiseedConfig"),
    "SeqLenScalingConfig": ("neuralops.benchmarks.config", "SeqLenScalingConfig"),
    "RulerConfig": ("neuralops.benchmarks.config", "RulerConfig"),
    "GPT2SweepConfig": ("neuralops.benchmarks.config", "GPT2SweepConfig"),
    "KDTestConfig": ("neuralops.benchmarks.config", "KDTestConfig"),
    "LMEvalAdapter": ("neuralops.benchmarks.eval_suite", "LMEvalAdapter"),
    "RULERBenchmark": ("neuralops.benchmarks.eval_suite", "RULERBenchmark"),
    "CustomEvaluationSuite": ("neuralops.benchmarks.eval_suite", "CustomEvaluationSuite"),
    "run_quality_sweep": ("neuralops.benchmarks.run_quality_sweep", "run_quality_sweep"),
    "run_gpt2_sweep": ("neuralops.benchmarks.run_gpt2_sweep", "run_gpt2_sweep"),
    "BenchmarkRunner": ("neuralops.benchmarks.harness", "BenchmarkRunner"),
    "BenchmarkMetric": ("neuralops.benchmarks.harness", "BenchmarkMetric"),
    "HardwareProfile": ("neuralops.benchmarks.harness", "HardwareProfile"),
}


def __getattr__(name: str) -> Any:
    """PEP 562 lazy attribute loading for neuralops.benchmarks symbols."""
    if name in _LAZY_BENCHMARK_SYMBOLS:
        mod_name, attr_name = _LAZY_BENCHMARK_SYMBOLS[name]
        module = importlib.import_module(mod_name)
        val = getattr(module, attr_name)
        globals()[name] = val
        return val

    try:
        module = importlib.import_module(f"neuralops.benchmarks.{name}")
        globals()[name] = module
        return module
    except ModuleNotFoundError:
        pass

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
