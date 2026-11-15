#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for MultiFormatSaver, concurrent multi-writer safety, and save_all_formats deprecation.
"""

from __future__ import annotations

import concurrent.futures
import sqlite3
import warnings
from pathlib import Path
import pytest

from neuralops.benchmarks.results.saver import MultiFormatSaver, save_all_formats
from neuralops.benchmarks.results.schema import BenchmarkResult, BenchmarkStatus


def test_multiformat_saver_all_targets(tmp_path: Path):
    """Verify MultiFormatSaver writes to JSON, journal, CSV, and SQLite."""
    out_dir = tmp_path / "results"
    saver = MultiFormatSaver(out_dir)

    result = BenchmarkResult(
        run_id="run_test_001",
        model_name="bert-base-uncased",
        transplanted_attention="test_attn",
        benchmark_suite="sst2",
        status="success",
        metrics={"accuracy": 0.92, "latency_ms": 15.4},
    )

    saved = saver.save(result)

    assert saved["raw_json"].exists()
    assert saved["journal"].exists()
    assert saved["summary_csv"].exists()
    assert saved["ledger_db"].exists()

    # Verify SQLite record
    conn = sqlite3.connect(str(saved["ledger_db"]))
    cur = conn.cursor()
    cur.execute("SELECT run_id, status, model_name FROM benchmark_runs WHERE run_id = ?", ("run_test_001",))
    row = cur.fetchone()
    conn.close()
    assert row is not None
    assert row[0] == "run_test_001"
    assert row[1] == BenchmarkStatus.SUCCESS.value
    assert row[2] == "bert-base-uncased"


def test_save_all_formats_delegation_and_warning(tmp_path: Path):
    """Verify save_all_formats raises DeprecationWarning and delegates to MultiFormatSaver."""
    out_dir = tmp_path / "results_dep"
    result = BenchmarkResult(
        run_id="run_dep_002",
        model_name="gpt2",
        transplanted_attention="rope_attn",
        benchmark_suite="wikitext2",
        status="success",
        metrics={"perplexity": 24.5},
    )

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        saved = save_all_formats(result, output_dir=out_dir)
        assert any("save_all_formats is deprecated" in str(item.message) for item in w)
        assert "raw_json" in saved
        assert saved["raw_json"].exists()


def test_multiformat_saver_concurrent_writers(tmp_path: Path):
    """Verify concurrent writes to the same directory do not conflict or corrupt."""
    out_dir = tmp_path / "results_concurrent"
    saver = MultiFormatSaver(out_dir)

    def write_worker(idx: int):
        r = BenchmarkResult(
            run_id=f"run_conc_{idx:03d}",
            model_name="bert",
            transplanted_attention=f"attn_{idx % 4}",
            benchmark_suite="glue",
            status="success",
            metrics={"accuracy": 0.80 + (idx * 0.001)},
        )
        saver.save(r)
        return idx

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(write_worker, i) for i in range(20)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    assert len(results) == 20

    # Check SQLite row count
    conn = sqlite3.connect(str(saver.ledger_db_path))
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM benchmark_runs")
    count = cur.fetchone()[0]
    conn.close()
    assert count == 20
