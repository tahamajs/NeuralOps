#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Modular benchmark suites for NeuralOps empirical evaluation.
"""

from __future__ import annotations

from typing import Dict, List, Type

from neuralops.benchmarks.core.base import AbstractBaseBenchmark
from neuralops.benchmarks.suites.attention import AttentionBenchmark
from neuralops.benchmarks.suites.compression import CompressionBenchmark
from neuralops.benchmarks.suites.cuda_ops import CudaOpsBenchmark
from neuralops.benchmarks.suites.e2e_serving import ServingBenchmark
from neuralops.benchmarks.suites.niah import NeedleInAHaystackBenchmark
from neuralops.benchmarks.suites.qwen_kd import QwenKDBenchmark
from neuralops.benchmarks.suites.roofline import RooflineBenchmark
from neuralops.benchmarks.suites.triton_ops import TritonOpsBenchmark

BENCHMARK_SUITES: Dict[str, Type[AbstractBaseBenchmark]] = {
    "attention": AttentionBenchmark,
    "compression": CompressionBenchmark,
    "cuda_ops": CudaOpsBenchmark,
    "e2e_serving": ServingBenchmark,
    "niah": NeedleInAHaystackBenchmark,
    "qwen_kd": QwenKDBenchmark,
    "roofline": RooflineBenchmark,
    "triton_ops": TritonOpsBenchmark,
}


def get_benchmark_suite(name: str) -> Type[AbstractBaseBenchmark]:
    clean_name = name.lower().strip()
    if clean_name not in BENCHMARK_SUITES:
        raise KeyError(f"Unknown benchmark suite '{name}'. Available suites: {list(BENCHMARK_SUITES.keys())}")
    return BENCHMARK_SUITES[clean_name]


def list_suites() -> List[str]:
    return sorted(list(BENCHMARK_SUITES.keys()))


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
