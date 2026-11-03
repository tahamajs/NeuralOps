#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for CLI error handling and exit codes."""

from neuralops.cli.core.errors import (
    ArchitectureMismatchError,
    BackendUnavailableError,
    ConfigurationError,
    ExitCode,
    NumericalGateError,
    RunDirectoryLockedError,
)


def test_exit_code_values() -> None:
    assert ExitCode.SUCCESS == 0
    assert ExitCode.INVALID_CONFIG == 2
    assert ExitCode.USAGE_ERROR == 2
    assert ExitCode.BACKEND_UNAVAILABLE == 3
    assert ExitCode.NUMERICAL_GATE_FAILED == 4
    assert ExitCode.ARCHITECTURE_MISMATCH == 5
    assert ExitCode.FILE_LOCK_FAILED == 6
    assert ExitCode.GATE_VERIFICATION_FAILED == 21
    assert ExitCode.MODULE_NOT_FOUND == 22
    assert ExitCode.RUN_RESUME_MISMATCH == 23
    assert ExitCode.INTERNAL_ERROR == 70


def test_custom_cli_exceptions() -> None:
    e1 = BackendUnavailableError("CUDA device not found", exit_code=ExitCode.BACKEND_UNAVAILABLE)
    assert e1.exit_code == ExitCode.BACKEND_UNAVAILABLE
    assert "CUDA device not found" in str(e1)

    e2 = ConfigurationError("Invalid key")
    assert e2.exit_code == ExitCode.INVALID_CONFIG

    e3 = NumericalGateError("Gate check failed")
    assert e3.exit_code == ExitCode.NUMERICAL_GATE_FAILED

    e4 = ArchitectureMismatchError("Layer incompatible")
    assert e4.exit_code == ExitCode.ARCHITECTURE_MISMATCH

    e5 = RunDirectoryLockedError("Locked by pid 1234")
    assert e5.exit_code == ExitCode.FILE_LOCK_FAILED
