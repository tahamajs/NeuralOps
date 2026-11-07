#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for Autonomous Circuit Breaker state machine and recovery."""

from __future__ import annotations

import time
import pytest
import torch

from neuralops.runtime.circuit_breaker import (
    AutonomousCircuitBreaker,
    CircuitBreakerFSM,
    CircuitState,
)


def test_circuit_state_enum_values():
    assert CircuitState.NORMAL == "Normal"
    assert CircuitState.DEGRADED == "Degraded"
    assert CircuitState.TRIPPED == "Tripped"


def test_circuit_breaker_fsm_transitions_and_trip():
    fsm = CircuitBreakerFSM(cooldown_seconds=0.050)
    assert fsm.state == CircuitState.NORMAL

    # First failure: NORMAL -> DEGRADED
    state = fsm.record_failure("CUDA kernel failure")
    assert state == CircuitState.DEGRADED
    assert fsm.state == CircuitState.DEGRADED
    assert fsm.failure_count == 1
    assert fsm.total_trips == 0

    # Second failure: DEGRADED -> TRIPPED
    state = fsm.record_failure("Triton JIT compilation error")
    assert state == CircuitState.TRIPPED
    assert fsm.state == CircuitState.TRIPPED
    assert fsm.failure_count == 2
    assert fsm.total_trips == 1


def test_circuit_breaker_fsm_cooldown_recovery():
    fsm = CircuitBreakerFSM(cooldown_seconds=0.020)
    fsm.record_failure("Failure 1")
    fsm.record_failure("Failure 2")
    assert fsm.state == CircuitState.TRIPPED

    # Immediately checking recovery should not recover if cooldown has not elapsed
    fsm.check_recovery()
    assert fsm.state == CircuitState.TRIPPED

    # Sleep past cooldown period
    time.sleep(0.025)
    fsm.check_recovery()
    assert fsm.state == CircuitState.DEGRADED

    # Sleep past cooldown period again to return to NORMAL
    time.sleep(0.025)
    fsm.check_recovery()
    assert fsm.state == CircuitState.NORMAL
    assert fsm.failure_count == 0


def test_circuit_breaker_force_reset():
    fsm = CircuitBreakerFSM()
    fsm.record_failure("Failure 1")
    fsm.record_failure("Failure 2")
    assert fsm.state == CircuitState.TRIPPED
    assert fsm.failure_count == 2

    fsm.force_reset()
    assert fsm.state == CircuitState.NORMAL
    assert fsm.failure_count == 0


def test_autonomous_circuit_breaker_execution_fallback():
    cb = AutonomousCircuitBreaker(cooldown_seconds=0.020)

    t1 = torch.tensor([1.0, 2.0])
    t2 = torch.tensor([3.0, 4.0])
    t3 = torch.tensor([5.0, 6.0])

    def lvl1(x):
        return x + t1

    def lvl2(x):
        return x + t2

    def lvl3(x):
        return x + t3

    inp = torch.zeros(2)

    # 1. Normal execution on Tier 1
    out = cb.execute(lvl1, lvl2, lvl3, inp)
    assert torch.allclose(out, t1)
    assert cb.level1_invocations == 1
    assert cb.level2_invocations == 0
    assert cb.level3_invocations == 0
    assert cb.fsm.state == CircuitState.NORMAL

    # 2. Tier 1 fails, gracefully degrades to Tier 2
    def failing_lvl1(x):
        raise RuntimeError("Kernel panic in Level 1")

    out2 = cb.execute(failing_lvl1, lvl2, lvl3, inp)
    assert torch.allclose(out2, t2)
    assert cb.level1_invocations == 2
    assert cb.level2_invocations == 1
    assert cb.level3_invocations == 0
    assert cb.fsm.state == CircuitState.DEGRADED

    # 3. Next execution while DEGRADED: Tier 2 is called. If Tier 2 fails, trips to Tier 3
    def failing_lvl2(x):
        raise RuntimeError("Triton JIT error in Level 2")

    out3 = cb.execute(lvl1, failing_lvl2, lvl3, inp)
    assert torch.allclose(out3, t3)
    assert cb.level1_invocations == 2
    assert cb.level2_invocations == 2
    assert cb.level3_invocations == 1
    assert cb.fsm.state == CircuitState.TRIPPED

    # 4. Next execution while TRIPPED: directly routes to Tier 3 (safe reference)
    out4 = cb.execute(lvl1, lvl2, lvl3, inp)
    assert torch.allclose(out4, t3)
    assert cb.level3_invocations == 2

    # 5. Recovery after cooldown: TRIPPED -> DEGRADED -> NORMAL
    time.sleep(0.025)
    out5 = cb.execute(lvl1, lvl2, lvl3, inp)
    # Recovered to DEGRADED, executes lvl2
    assert torch.allclose(out5, t2)
    assert cb.fsm.state == CircuitState.DEGRADED

    time.sleep(0.025)
    out6 = cb.execute(lvl1, lvl2, lvl3, inp)
    # Recovered to NORMAL, executes lvl1
    assert torch.allclose(out6, t1)
    assert cb.fsm.state == CircuitState.NORMAL
