"""Evaluation metrics re-exported from trainer.metrics."""

from framework.trainer.metrics import (
    aggregate_results,
    compute_accuracy,
    compute_f1,
    compute_matthews_corrcoef,
    compute_pearson_spearman,
    compute_precision_recall,
    compute_regression_metrics,
    get_task_metrics,
)

__all__ = [
    "get_task_metrics",
    "compute_accuracy",
    "compute_f1",
    "compute_matthews_corrcoef",
    "compute_pearson_spearman",
    "compute_precision_recall",
    "compute_regression_metrics",
    "aggregate_results",
]
