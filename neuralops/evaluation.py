#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""NeuralOps evaluation package bridge.

Exposes the unified evaluation and benchmarking interface.
"""

from __future__ import annotations

from neuralops.benchmarks.eval_bridge import (
    BenchmarkEvalResult,
    BenchmarkEvaluator,
    create_evaluator,
)

__all__ = [
    "BenchmarkEvaluator",
    "BenchmarkEvalResult",
    "create_evaluator",
]
