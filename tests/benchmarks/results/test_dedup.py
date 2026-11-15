#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for Deduplicator index management and file locking.
"""

from __future__ import annotations

from pathlib import Path
import pytest

from neuralops.benchmarks.results.dedup import Deduplicator
from neuralops.benchmarks.results.schema import BenchmarkResult


def test_deduplicator_lifecycle(tmp_path: Path):
    """Verify deduplication checks and persistence with locking."""
    out_dir = tmp_path / "results"
    dedup = Deduplicator(out_dir)

    res = BenchmarkResult(
        run_id="run_dedup_01",
        module={"name": "mod_x"},
        model={"name": "bert"},
        dataset={"name": "sst2"},
        status="success",
        metrics={"accuracy": 0.88},
    )
    key = res.compute_run_key()

    should_skip, reason = dedup.should_skip(key)
    assert not should_skip

    dedup.record_run(res, out_dir / "raw_results" / "run_dedup_01.json")

    # Second check must indicate duplicate
    should_skip2, reason2 = dedup.should_skip(key)
    assert should_skip2
    assert "already exists" in reason2

    # Force rerun overrides
    should_skip_forced, _ = dedup.should_skip(key, force_rerun=True)
    assert not should_skip_forced


def test_deduplicator_failed_run_retry(tmp_path: Path):
    """Verify previously failed runs are allowed to retry."""
    out_dir = tmp_path / "results"
    dedup = Deduplicator(out_dir)

    res_fail = BenchmarkResult(
        run_id="run_dedup_failed",
        module={"name": "mod_fail"},
        model={"name": "bert"},
        dataset={"name": "sst2"},
        status="failed",
    )
    key = res_fail.compute_run_key()

    dedup.record_run(res_fail, out_dir / "raw_results" / "run_dedup_failed.json")

    should_skip, reason = dedup.should_skip(key)
    assert not should_skip
    assert "allowing retry" in reason
