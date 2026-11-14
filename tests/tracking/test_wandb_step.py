#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for step parameter passing in WandBTracker."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

from neuralops.tracking.backends import WandBTracker


def test_wandb_log_metric_passes_step_kwarg(monkeypatch):
    """Verify WandBTracker passes step=step as keyword argument and not inside data dict."""
    mock_wandb = MagicMock()
    mock_run = MagicMock()
    mock_wandb.init.return_value = mock_run
    monkeypatch.setitem(sys.modules, "wandb", mock_wandb)

    tracker = WandBTracker(project="test_proj", experiment_name="test_exp")

    # 1. log_metric with step
    tracker.log_metric("train/loss", 0.42, step=10)
    mock_run.log.assert_called_with({"train/loss": 0.42}, step=10)

    # 2. log_metric without step
    mock_run.log.reset_mock()
    tracker.log_metric("val/loss", 0.35)
    mock_run.log.assert_called_with({"val/loss": 0.35})

    # 3. log_metrics with step
    mock_run.log.reset_mock()
    tracker.log_metrics({"a": 1.0, "b": 2.0}, step=20)
    mock_run.log.assert_called_with({"a": 1.0, "b": 2.0}, step=20)

    # 4. log_metrics without step
    mock_run.log.reset_mock()
    tracker.log_metrics({"a": 1.0, "b": 2.0})
    mock_run.log.assert_called_with({"a": 1.0, "b": 2.0})
