#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Regression tests for metrics history serialization."""

import json

from neuralops.experiment_logging.tracker import MetricsTracker


def test_jsonl_path_writes_one_json_record_per_line(tmp_path):
    output = tmp_path / "run" / "metrics.jsonl"
    tracker = MetricsTracker(str(output))
    tracker.log({"loss": 0.5}, step=1)
    tracker.log({"loss": 0.25}, step=2)

    tracker.flush()

    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert [json.loads(line)["step"] for line in lines] == [1, 2]


def test_metrics_json_path_keeps_json_array_compatibility(tmp_path):
    output = tmp_path / "metrics.json"
    tracker = MetricsTracker(str(output))
    tracker.log({"note": "دقت"})

    tracker.save()

    assert json.loads(output.read_text(encoding="utf-8")) == [{"note": "دقت"}]


def test_metrics_tracker_supports_filename_without_parent_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tracker = MetricsTracker("metrics.jsonl")
    tracker.log({"loss": 1.0})

    tracker.save()

    assert json.loads((tmp_path / "metrics.jsonl").read_text(encoding="utf-8")) == {"loss": 1.0}
