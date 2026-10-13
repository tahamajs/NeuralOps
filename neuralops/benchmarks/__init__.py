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

# Auto-wired package members
from neuralops.benchmarks import adapters
from neuralops.benchmarks import core
from neuralops.benchmarks import data
from neuralops.benchmarks import metrics
from neuralops.benchmarks import models
from neuralops.benchmarks import results
from neuralops.benchmarks import utils
from neuralops.benchmarks import __main__
from neuralops.benchmarks import api
from neuralops.benchmarks import cli
from neuralops.benchmarks import config
from neuralops.benchmarks import discovery
from neuralops.benchmarks import eval_suite
from neuralops.benchmarks import executor
from neuralops.benchmarks import gpu_live_probe
from neuralops.benchmarks import harvest_gpu_benchmark_data
from neuralops.benchmarks import kd_quick_test
from neuralops.benchmarks import kernel_benchmark
from neuralops.benchmarks import launch_parallel_sweep
from neuralops.benchmarks import logging
from neuralops.benchmarks import multi_mode_benchmark
from neuralops.benchmarks import orchestrator
from neuralops.benchmarks import registry
from neuralops.benchmarks import result
from neuralops.benchmarks import run_all_benchmarks
from neuralops.benchmarks import run_all_modules_scheduler
from neuralops.benchmarks import run_gpt2_sweep
from neuralops.benchmarks import run_long_context_sweep
from neuralops.benchmarks import run_multiseed
from neuralops.benchmarks import run_multiseed_rigor
from neuralops.benchmarks import run_one_module_isolated
from neuralops.benchmarks import run_quality_sweep
from neuralops.benchmarks import run_rank_sweeper
from neuralops.benchmarks import run_ruler_benchmark
from neuralops.benchmarks import run_seqlen_scaling
from neuralops.benchmarks import run_seqlen_scaling_bench
from neuralops.benchmarks import run_whitened_svd
from neuralops.benchmarks import runner
from neuralops.benchmarks import suite_runner
from neuralops.benchmarks import writers
