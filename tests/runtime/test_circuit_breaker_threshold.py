#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for circuit breaker failure threshold configuration."""

from __future__ import annotations

from neuralops.runtime.circuit_breaker import CircuitBreakerFSM, CircuitState


def test_circuit_breaker_respects_failure_threshold():
    """Verify that state transition occurs only when failure_count reaches failure_threshold."""
    fsm = CircuitBreakerFSM(cooldown_seconds=1.0, failure_threshold=3)
    assert fsm.state == CircuitState.NORMAL
    assert fsm.failure_count == 0

    # Failure 1: Still NORMAL
    fsm.record_failure("error 1")
    assert fsm.state == CircuitState.NORMAL
    assert fsm.failure_count == 1

    # Failure 2: Still NORMAL
    fsm.record_failure("error 2")
    assert fsm.state == CircuitState.NORMAL
    assert fsm.failure_count == 2

    # Failure 3: Transitions to DEGRADED and resets failure_count
    fsm.record_failure("error 3")
    assert fsm.state == CircuitState.DEGRADED
    assert fsm.failure_count == 0

    # Next failures towards TRIPPED
    fsm.record_failure("error 4")
    assert fsm.state == CircuitState.DEGRADED
    assert fsm.failure_count == 1

    fsm.record_failure("error 5")
    assert fsm.state == CircuitState.DEGRADED
    assert fsm.failure_count == 2

    fsm.record_failure("error 6")
    assert fsm.state == CircuitState.TRIPPED
    assert fsm.failure_count == 0
