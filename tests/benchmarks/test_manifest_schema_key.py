#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests verifying benchmark manifest schema key naming convention."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from neuralops.benchmarks.core.base import ExecutionSummary
from neuralops.benchmarks.core.runner import BenchmarkRunner
from neuralops.config.benchmark_schema import BenchmarkRunConfig


def test_manifest_uses_dollar_schema_key(tmp_path: Path) -> None:
    """Verify that BenchmarkRunner writes the schema URL under '$schema', never under an empty key ''."""
    cfg = BenchmarkRunConfig(
        name="test_manifest_schema",
        output_root_directory=str(tmp_path),
        benchmark_suite="attention",
    )

    runner = BenchmarkRunner(cfg)

    dummy_summary = ExecutionSummary(
        run_id=runner.paths.run_id,
        benchmark_name=cfg.benchmark_suite,
        status="COMPLETED",
        total_iterations=1,
        passed_iterations=1,
    )
    mock_suite = MagicMock()
    mock_suite.execute.return_value = dummy_summary

    with patch("neuralops.benchmarks.core.runner.get_benchmark_suite", return_value=lambda *a, **k: mock_suite):
        summary = runner.run()

    assert summary.status == "COMPLETED"
    manifest_path = runner.paths.manifest_json
    assert manifest_path.exists(), f"Manifest file {manifest_path} was not created"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    assert "$schema" in manifest_data, f"Expected '$schema' key in manifest, keys found: {list(manifest_data.keys())}"
    assert "" not in manifest_data, f"Empty string key '' must not exist in manifest, keys found: {list(manifest_data.keys())}"
    assert manifest_data["$schema"] == "https://raw.githubusercontent.com/tahamajs/NeuralOps/main/schemas/benchmark_manifest_v1.json"
