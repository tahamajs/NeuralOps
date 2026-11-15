#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for GitCommitter automated results versioning.
"""

from __future__ import annotations

from pathlib import Path
import pytest

from neuralops.benchmarks.results.git_sync import GitCommitter
from neuralops.benchmarks.results.schema import BenchmarkResult


def test_build_commit_message():
    """Verify structured commit message generation."""
    committer = GitCommitter()
    result = BenchmarkResult(
        run_id="run_git_001",
        module={"name": "flash_attn"},
        model={"name": "bert"},
        dataset={"name": "sst2"},
        status="success",
        metrics={"accuracy": 0.932},
    )

    msg = committer.build_commit_message(result, completed_count=5, failed_count=0)
    assert "results: flash_attn on bert/sst2" in msg
    assert "acc=93.2%" in msg
    assert "5 runs completed" in msg


def test_git_committer_dry_run(tmp_path: Path):
    """Verify register_run with dry_run=True returns dry_run indicator."""
    committer = GitCommitter(
        output_dir=tmp_path / "results",
        auto_commit=True,
        commit_interval=1,
        dry_run=True,
    )
    result = BenchmarkResult(
        run_id="run_git_002",
        status="success",
    )
    res_file = tmp_path / "results" / "raw_results" / "run_git_002.json"
    res_file.parent.mkdir(parents=True, exist_ok=True)
    res_file.write_text("{}", encoding="utf-8")

    out = committer.register_run(result, {"json": res_file})
    assert out == "dry_run"
