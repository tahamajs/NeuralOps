#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Unit tests verifying --benchmark CLI argument resolution and enforcement."""

from __future__ import annotations

import argparse
from unittest.mock import MagicMock, patch
import pytest

from neuralops.benchmarks.cli import _build_parser, handle_run


def test_handle_run_respects_benchmark_flag():
    """Assert handle_run runs ONLY the benchmark specified via --benchmark."""
    parser = _build_parser()
    args = parser.parse_args(["run", "--benchmark", "attention:cosformer", "--device", "cpu"])

    mock_orchestrator = MagicMock()
    mock_orchestrator.run_registered_ops.return_value = [{"status": "PASS"}]

    with patch("neuralops.benchmarks.orchestrator.BenchmarkOrchestrator", return_value=mock_orchestrator) as mock_orch_cls, \
         patch("neuralops.benchmarks.api.Runner") as mock_runner_cls:
        handle_run(args)

        # Confirm either orchestrator or Runner was invoked with exactly the selected benchmark, NOT 'all'
        if mock_orch_cls.called:
            kwargs = mock_orch_cls.call_args.kwargs
            modules_arg = kwargs.get("modules")
            assert modules_arg != "all", "Orchestrator called with modules='all' instead of requested benchmark"
            assert "cosformer" in modules_arg
        elif mock_runner_cls.called:
            kwargs = mock_runner_cls.call_args.kwargs
            modules_arg = kwargs.get("modules")
            assert modules_arg != "all", "Runner called with modules='all' instead of requested benchmark"
            assert modules_arg == ["cosformer"] or modules_arg == ["attention:cosformer"]
        else:
            pytest.fail("Neither BenchmarkOrchestrator nor Runner was called")


def test_handle_run_invalid_benchmark_fails():
    """Assert handle_run raises ValueError on invalid benchmark and lists valid names."""
    parser = _build_parser()
    args = parser.parse_args(["run", "--benchmark", "invalid_nonexistent_benchmark_xyz", "--device", "cpu"])

    with patch("neuralops.benchmarks.orchestrator.BenchmarkOrchestrator") as mock_orch, \
         patch("neuralops.benchmarks.api.Runner") as mock_runner:
        with pytest.raises(ValueError) as exc_info:
            handle_run(args)

        msg = str(exc_info.value)
        assert "invalid_nonexistent_benchmark_xyz" in msg
        assert "Valid benchmark" in msg
        assert not mock_orch.called
        assert not mock_runner.called


def test_dispatch_benchmark_subcommands():
    """Assert _dispatch cleanly routes neuralops benchmark <action> (info, list, plan, run)."""
    from neuralops.benchmarks.cli import _dispatch

    with patch("neuralops.cli.info.cmd_info", return_value=0) as mock_info:
        args = argparse.Namespace(subcommand="benchmark", action="info")
        assert _dispatch(args) == 0
        assert mock_info.called

    with patch("neuralops.benchmarks.cli.handle_list", return_value=0) as mock_list:
        args = argparse.Namespace(subcommand="benchmark", action="list")
        assert _dispatch(args) == 0
        assert mock_list.called

    with patch("neuralops.benchmarks.cli.handle_plan") as mock_plan:
        args = argparse.Namespace(subcommand="benchmark", action="plan")
        assert _dispatch(args) == 0
        assert mock_plan.called

    with patch("neuralops.benchmarks.cli.handle_run") as mock_run:
        args = argparse.Namespace(subcommand="benchmark", action="run")
        assert _dispatch(args) == 0
        assert mock_run.called


def test_compare_subcommand(tmp_path):
    """Assert compare subcommand invokes compare_results and reflects regression gate status."""
    import json
    from neuralops.benchmarks.cli import _build_parser, _dispatch

    base_file = tmp_path / "base.json"
    cand_file = tmp_path / "cand.json"
    base_file.write_text(json.dumps({"metrics": {"latency": 10.0}, "run_id": "r1"}))
    cand_file.write_text(json.dumps({"metrics": {"latency": 10.5}, "run_id": "r2"}))

    parser = _build_parser()
    args_pass = parser.parse_args([
        "compare",
        "--baseline", str(base_file),
        "--candidate", str(cand_file),
        "--metric", "latency",
        "--max-regression", "1.0",
    ])
    assert _dispatch(args_pass) == 0

    args_fail = parser.parse_args([
        "compare",
        "--baseline", str(base_file),
        "--candidate", str(cand_file),
        "--metric", "latency",
        "--max-regression", "0.1",
    ])
    assert _dispatch(args_fail) == 1
