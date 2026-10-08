"""neuralops benchmarking suite."""
from __future__ import annotations

from neuralops.benchmarks.api import Runner, benchmark
from neuralops.benchmarks.cli import main
from neuralops.benchmarks.config import (
    BenchmarkConfig,
    BenchmarkRunConfig,
    GPT2SweepConfig,
    KDTestConfig,
    MultiModeConfig,
    MultiseedConfig,
    QualitySweepConfig,
    RankSweepConfig,
    RulerConfig,
    SeqLenScalingConfig,
)
from neuralops.benchmarks.result import BenchmarkResult

from neuralops.benchmarks.eval_suite import (
    LMEvalAdapter,
    RULERBenchmark,
    CustomEvaluationSuite,
)

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
]
