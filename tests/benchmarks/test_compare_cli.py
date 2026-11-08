#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for compare CLI subcommand and benchmark alias resolution (AGT-59 / Gaps 2 & 6).

Verifies that:
  1. `neuralops benchmark list` succeeds without requiring CUDA when CUDA is unavailable.
  2. `neuralops benchmark info <name>` inspects module info and does not fall into `handle_run`.
  3. `neuralops compare` exits 0 when candidate result is within tolerance.
  4. `neuralops compare` exits 4 (FAIL_NUMERICS) when regression exceeds threshold.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
import pytest

from neuralops.benchmarks.cli import _build_parser, _dispatch


def test_benchmark_list_does_not_hit_cuda():
    """`neuralops benchmark list` must work on CPU-only hosts without touching CUDA."""
    parser = _build_parser()
    args = parser.parse_args(["benchmark", "list"])

    with patch("torch.cuda.is_available", return_value=False), \
         patch("neuralops.benchmarks.cli.handle_run") as mock_run:
        exit_code = _dispatch(args)
        assert exit_code == 0
        assert not mock_run.called, "benchmark list must not route to handle_run"


def test_benchmark_alias_no_longer_runs_all():
    """`benchmark info <name>` must route to info/inspect and never fall into handle_run."""
    parser = _build_parser()
    args = parser.parse_args(["benchmark", "info", "attention:cosformer"])

    with patch("neuralops.benchmarks.cli.handle_run") as mock_run:
        exit_code = _dispatch(args)
        assert exit_code == 0
        assert not mock_run.called, "benchmark info <name> must never fall into handle_run"


def test_compare_within_tolerance_exits_0(tmp_path):
    """`neuralops compare` exits 0 when regression is within specified tolerance."""
    base_file = tmp_path / "base.json"
    cand_file = tmp_path / "cand.json"
    base_file.write_text(json.dumps({"metrics": {"latency_ms": 10.0}, "run_id": "r1"}))
    cand_file.write_text(json.dumps({"metrics": {"latency_ms": 10.2}, "run_id": "r2"}))

    parser = _build_parser()
    args = parser.parse_args([
        "compare",
        "--baseline", str(base_file),
        "--candidate", str(cand_file),
        "--metric", "latency_ms",
        "--max-regression-pct", "0.5",
    ])
    exit_code = _dispatch(args)
    assert exit_code == 0


def test_compare_regression_exits_4(tmp_path):
    """`neuralops compare` exits 4 (FAIL_NUMERICS) when regression exceeds tolerance threshold."""
    base_file = tmp_path / "base.json"
    cand_file = tmp_path / "cand.json"
    base_file.write_text(json.dumps({"metrics": {"latency_ms": 10.0}, "run_id": "r1"}))
    cand_file.write_text(json.dumps({"metrics": {"latency_ms": 15.0}, "run_id": "r2"}))

    parser = _build_parser()
    args = parser.parse_args([
        "compare",
        "--baseline", str(base_file),
        "--candidate", str(cand_file),
        "--metric", "latency_ms",
        "--max-regression-pct", "0.5",
    ])
    exit_code = _dispatch(args)
    assert exit_code == 4
