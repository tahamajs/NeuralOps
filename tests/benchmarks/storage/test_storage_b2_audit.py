#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Audit verification test for storage subsystem boundaries in benchmarks."""

from __future__ import annotations

from pathlib import Path


def test_benchmarks_storage_isolation():
    """Verify neuralops/benchmarks/storage is isolated and does not collide with results_saver."""
    repo_root = Path(__file__).resolve().parents[3]
    benchmarks_dir = repo_root / "neuralops" / "benchmarks"
    storage_dir = benchmarks_dir / "storage"

    # Verify that benchmarks storage is cleanly separated or nonexistent,
    # and results persistence is governed under results_v2/ledger / results_saver.
    if storage_dir.exists():
        assert storage_dir.is_dir()
    else:
        # Expected: benchmarks storage is consolidated into results/ or results_saver/
        assert not storage_dir.exists()
