"""Tests for neuralops.analysis.metrics module."""

import json
from pathlib import Path
import numpy as np
import pytest

from neuralops.analysis.metrics import (
    MetricsTracker,
    aggregate_results,
    compute_metrics,
)


class TestMetrics:
    """Test metrics computation and result aggregation."""

    def test_compute_metrics_classification(self):
        # 4 samples, 2 classes
        preds = np.array([[0.8, 0.2], [0.1, 0.9], [0.7, 0.3], [0.3, 0.7]])
        labels = np.array([0, 1, 0, 1])
        res = compute_metrics(preds, labels, task="classification")
        assert "accuracy" in res
        assert res["accuracy"] == 1.0
        assert "f1_macro" in res
        assert "matthews_corrcoef" in res

    def test_compute_metrics_regression(self):
        preds = np.array([1.0, 2.0, 3.0, 4.0])
        labels = np.array([1.1, 1.9, 3.2, 3.9])
        res = compute_metrics(preds, labels, task="regression")
        assert "pearson" in res
        assert "spearman" in res
        assert res["pearson"] > 0.9

    def test_aggregate_results(self):
        records = [
            {"accuracy": 0.80, "loss": 0.5},
            {"accuracy": 0.90, "loss": 0.3},
        ]
        agg = aggregate_results(records)
        assert agg["accuracy_mean"] == pytest.approx(0.85)
        assert agg["loss_mean"] == pytest.approx(0.4)
        assert len(agg["accuracy_values"]) == 2


class TestMetricsTracker:
    """Test step metric logging and serialization via JSON lines."""

    def test_tracker_logging_and_flush(self, tmp_path):
        log_file = tmp_path / "metrics.jsonl"
        tracker = MetricsTracker(str(log_file))

        tracker.log({"loss": 0.5, "acc": 0.8}, step=1)
        tracker.log({"loss": 0.3, "acc": 0.9}, step=2, flush=True)

        assert log_file.exists()
        df = tracker.to_dataframe()
        assert len(df) == 2
        assert list(df["step"]) == [1, 2]
