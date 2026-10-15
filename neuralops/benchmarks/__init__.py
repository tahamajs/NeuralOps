"""neuralops benchmarking suite."""
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
