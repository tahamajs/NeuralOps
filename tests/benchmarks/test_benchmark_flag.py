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
