#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for avoiding nested runs in MLflowTracker."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

from neuralops.tracking.backends import MLflowTracker


def test_mlflow_creates_new_run_when_no_active(monkeypatch):
    """Verify MLflowTracker creates a new run and ends it on close when none active."""
    mock_mlflow = MagicMock()
    mock_mlflow.active_run.return_value = None
    new_run = MagicMock()
    mock_mlflow.start_run.return_value = new_run
    monkeypatch.setitem(sys.modules, "mlflow", mock_mlflow)

    tracker = MLflowTracker(experiment_name="test_exp")
    assert tracker.run is new_run
    assert tracker._created_here is True
    mock_mlflow.start_run.assert_called_once()

    tracker.close()
    mock_mlflow.end_run.assert_called_once()


def test_mlflow_reuses_existing_active_run(monkeypatch, caplog):
    """Verify MLflowTracker reuses active run and does NOT end it on close."""
    mock_mlflow = MagicMock()
    existing_run = MagicMock()
    existing_run.info.run_id = "run_12345"
    mock_mlflow.active_run.return_value = existing_run
    monkeypatch.setitem(sys.modules, "mlflow", mock_mlflow)

    tracker = MLflowTracker(experiment_name="test_exp")
    assert tracker.run is existing_run
    assert tracker._created_here is False
    mock_mlflow.start_run.assert_not_called()

    # Warning logged
    assert "Reusing existing active MLflow run: run_12345" in caplog.text

    tracker.close()
    mock_mlflow.end_run.assert_not_called()
