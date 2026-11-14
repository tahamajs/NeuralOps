#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for bounded memory in LocalTracker."""

from __future__ import annotations

import json
from pathlib import Path

from neuralops.tracking.backends import LocalTracker


def test_local_tracker_bounded_memory_default(tmp_path: Path):
    """Verify LocalTracker defaults to keep_in_memory=False and does not leak memory."""
    tracker = LocalTracker(log_dir=str(tmp_path))
    assert not tracker._keep_in_memory

    for i in range(100):
        tracker.log_metric("loss", 0.5, step=i)

    assert len(tracker.metrics) == 0, "metrics list must remain empty when keep_in_memory is False"

    # Verify metrics were written to disk
    metrics_file = tmp_path / "metrics.jsonl"
    assert metrics_file.exists()
    lines = [json.loads(line) for line in metrics_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 100
    assert lines[0]["name"] == "loss"
    assert lines[0]["step"] == 0


def test_local_tracker_keep_in_memory_true(tmp_path: Path):
    """Verify LocalTracker retains metrics when keep_in_memory=True."""
    tracker = LocalTracker(log_dir=str(tmp_path), keep_in_memory=True)
    assert tracker._keep_in_memory

    for i in range(10):
        tracker.log_metric("accuracy", 0.95, step=i)

    assert len(tracker.metrics) == 10
    assert tracker.metrics[0]["name"] == "accuracy"
    assert tracker.metrics[0]["value"] == 0.95
