# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
NeuralOps Benchmarks Metrics Subpackage.
"""

from neuralops.benchmarks.metrics.evaluator import (
    ClassificationMetrics,
    LanguageModelingMetrics,
    METRIC_REGISTRY,
    Metric,
    QuestionAnsweringMetrics,
    RegressionMetrics,
    SummarizationMetrics,
    compute_task_metrics,
    register_metric,
)

__all__ = [
    "Metric",
    "register_metric",
    "ClassificationMetrics",
    "RegressionMetrics",
    "LanguageModelingMetrics",
    "QuestionAnsweringMetrics",
    "SummarizationMetrics",
    "compute_task_metrics",
    "METRIC_REGISTRY",
]

from neuralops.benchmarks.metrics import evaluator
