"""Backward-compatibility shim for neuralops.benchmarks.suites.

This package is deprecated and will be removed in a future release.
Please use neuralops.benchmarks.suites_v2 instead.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "neuralops.benchmarks.suites is deprecated and will be removed in a future release. "
    "Please use neuralops.benchmarks.suites_v2 instead.",
    DeprecationWarning,
    stacklevel=2,
)

from neuralops.benchmarks.suites_v2 import (
    AbstractBaseBenchmark,
    AttentionBenchmark,
    BENCHMARK_SUITES,
    CompressionBenchmark,
    CudaOpsBenchmark,
    NeedleInAHaystackBenchmark,
    QwenKDBenchmark,
    RooflineBenchmark,
    ServingBenchmark,
    TritonOpsBenchmark,
    get_benchmark_suite,
    list_suites,
)

__all__ = [
    "AbstractBaseBenchmark",
    "AttentionBenchmark",
    "CompressionBenchmark",
    "CudaOpsBenchmark",
    "NeedleInAHaystackBenchmark",
    "QwenKDBenchmark",
    "RooflineBenchmark",
    "ServingBenchmark",
    "TritonOpsBenchmark",
    "BENCHMARK_SUITES",
    "get_benchmark_suite",
    "list_suites",
]
