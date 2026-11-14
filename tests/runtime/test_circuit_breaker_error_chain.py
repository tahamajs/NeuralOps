#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for error chaining and cascade history when all circuit breaker levels fail."""

from __future__ import annotations

import pytest
import torch

from neuralops.runtime.circuit_breaker import AutonomousCircuitBreaker


def test_circuit_breaker_all_fail_error_chain():
    """Verify RuntimeError details all level errors and is chained from level 3 exc."""
    cb = AutonomousCircuitBreaker(cooldown_seconds=10.0)

    def lvl1(x):
        raise ValueError("CUDA C++ kernel fault")

    def lvl2(x):
        raise TypeError("Triton compilation bug")

    def lvl3(x):
        raise ZeroDivisionError("PyTorch reference failure")

    inp = torch.zeros(2)
    with pytest.raises(RuntimeError) as exc_info:
        cb.execute(lvl1, lvl2, lvl3, inp)

    err_msg = str(exc_info.value)
    assert "All circuit-breaker levels failed." in err_msg
    assert "Level 1: ValueError: CUDA C++ kernel fault" in err_msg
    assert "Level 2: TypeError: Triton compilation bug" in err_msg
    assert "Level 3: ZeroDivisionError: PyTorch reference failure" in err_msg

    # Verify exception chaining (__cause__)
    assert isinstance(exc_info.value.__cause__, ZeroDivisionError)
    assert str(exc_info.value.__cause__) == "PyTorch reference failure"
