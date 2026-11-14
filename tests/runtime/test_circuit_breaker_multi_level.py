#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for multi-level cooldown recovery in CircuitBreakerFSM."""

from __future__ import annotations

import time

from neuralops.runtime.circuit_breaker import CircuitBreakerFSM, CircuitState


def test_multi_level_recovery_tripped_to_normal():
    """Verify that if enough cooldown periods elapse, TRIPPED recovers directly to NORMAL in one call."""
    cooldown = 0.020
    fsm = CircuitBreakerFSM(cooldown_seconds=cooldown)
    fsm.record_failure("error 1")
    fsm.record_failure("error 2")
    assert fsm.state == CircuitState.TRIPPED

    # Sleep past 2 full cooldown cycles (e.g. 50ms >= 2 * 20ms = 40ms)
    time.sleep(0.050)
    recovered_state = fsm.check_recovery()
    assert recovered_state == CircuitState.NORMAL
    assert fsm.state == CircuitState.NORMAL
    assert fsm.failure_count == 0


def test_single_level_recovery_when_only_one_period_elapses():
    """Verify that if only 1 cooldown period elapses, TRIPPED recovers only to DEGRADED."""
    cooldown = 0.040
    fsm = CircuitBreakerFSM(cooldown_seconds=cooldown)
    fsm.record_failure("error 1")
    fsm.record_failure("error 2")
    assert fsm.state == CircuitState.TRIPPED

    # Sleep ~45ms (1 cooldown cycle, but less than 2 cycles)
    time.sleep(0.045)
    recovered_state = fsm.check_recovery()
    assert recovered_state == CircuitState.DEGRADED
    assert fsm.state == CircuitState.DEGRADED
