#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for ResultsIndex query engine and error handling.
"""

from __future__ import annotations

import logging
from pathlib import Path
import pytest

from neuralops.benchmarks.results.index import ResultsIndex
from neuralops.benchmarks.results.saver import MultiFormatSaver
from neuralops.benchmarks.results.schema import BenchmarkResult


def test_index_best_unknown_metric_raises(tmp_path: Path):
    """Verify _best_from_list raises ValueError on unknown metric."""
    out_dir = tmp_path / "results"
    saver = MultiFormatSaver(out_dir)

    r = BenchmarkResult(
        run_id="run_idx_01",
        module={"name": "test_mod"},
        model={"name": "test_model"},
        dataset={"name": "test_ds"},
        status="success",
        metrics={"accuracy": 0.85},
    )
    saver.save(r)

    idx = ResultsIndex(out_dir)
    with pytest.raises(ValueError, match="not found in any results"):
        idx.best(metric="completely_unknown_metric_abc")


def test_index_best_and_find(tmp_path: Path):
    """Verify find() and best() operate accurately across saved results."""
    out_dir = tmp_path / "results"
    saver = MultiFormatSaver(out_dir)

    r1 = BenchmarkResult(
        run_id="run_1",
        module={"name": "mod_a"},
        model={"name": "bert"},
        dataset={"name": "sst2"},
        status="success",
        metrics={"accuracy": 0.82, "latency_ms": 10.0},
    )
    r2 = BenchmarkResult(
        run_id="run_2",
        module={"name": "mod_b"},
        model={"name": "bert"},
        dataset={"name": "sst2"},
        status="success",
        metrics={"accuracy": 0.94, "latency_ms": 15.0},
    )
    saver.save(r1)
    saver.save(r2)

    idx = ResultsIndex(out_dir)
    results = idx.find(model="bert")
    assert len(results) == 2

    best_acc = idx.best(metric="accuracy")
    assert best_acc is not None
    assert (best_acc.module.get("name") if isinstance(best_acc.module, dict) else best_acc.module) == "mod_b"


def test_index_corrupt_file_logs_warning(tmp_path: Path, caplog):
    """Verify _load_results logs warnings when encountering corrupt json."""
    out_dir = tmp_path / "results"
    raw_dir = out_dir / "raw_results"
    raw_dir.mkdir(parents=True, exist_ok=True)
    corrupt_json = raw_dir / "corrupt_run.json"
    corrupt_json.write_text("NOT_VALID_JSON{", encoding="utf-8")

    idx = ResultsIndex(out_dir)
    with caplog.at_level(logging.WARNING):
        res = idx._load_results(force_reload=True)

    assert any("Failed to parse benchmark result" in record.message for record in caplog.records)
