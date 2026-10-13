"""Metrics subsystem package."""

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

# Auto-wired package members
from neuralops.benchmarks.metrics import evaluator
