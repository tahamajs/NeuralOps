#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for examples/34_circuit_breaker_runtime_resilience.py.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import torch

from neuralops.runtime.circuit_breaker import (
    AutonomousCircuitBreaker,
    CircuitBreakerFSM,
    CircuitState,
)

EXAMPLE_PATH = Path(__file__).resolve().parent.parent.parent / "examples" / "34_circuit_breaker_runtime_resilience.py"


def test_34_runs_clean_cpu():
    """Verify Example 34 runs cleanly via subprocess with exit code 0."""
    proc = subprocess.run(
        [sys.executable, str(EXAMPLE_PATH), "--device", "cpu", "--cooldown-ms", "50"],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, f"Example 34 failed with stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"
    assert "[PASS] Overall Example 34 Status" in proc.stdout
    assert "Tier 1 Chaos Handled -> Degraded to Tier 2" in proc.stdout
    assert "Tier 2 Chaos Handled -> Tripped to Tier 3" in proc.stdout
    assert "Full Health Restored" in proc.stdout


def test_34_circuit_breaker_state_transitions():
    """Verify AutonomousCircuitBreaker FSM state transitions: NORMAL -> DEGRADED -> TRIPPED -> recovery."""
    cb = AutonomousCircuitBreaker(cooldown_seconds=0.02, enable_oom_healing=True)
    assert cb.fsm.state == CircuitState.NORMAL

    x = torch.tensor([1.0, 2.0])

    def l1_panic(t):
        raise RuntimeError("Level 1 Kernel Panic")

    def l2_panic(t):
        raise RuntimeError("Level 2 Triton Error")

    def l2_healthy(t):
        return t * 2.0

    def l3_reference(t):
        return t * 3.0

    # Trigger Level 1 failure -> DEGRADED
    out1 = cb.execute(l1_panic, l2_healthy, l3_reference, x)
    assert torch.allclose(out1, x * 2.0)
    assert cb.fsm.state == CircuitState.DEGRADED

    # Trigger Level 2 failure -> TRIPPED
    out2 = cb.execute(l1_panic, l2_panic, l3_reference, x)
    assert torch.allclose(out2, x * 3.0)
    assert cb.fsm.state == CircuitState.TRIPPED

    # Cooldown recovery step 1: TRIPPED -> DEGRADED
    time.sleep(0.03)
    recovered_state1 = cb.fsm.check_recovery()
    assert recovered_state1 == CircuitState.DEGRADED

    # Cooldown recovery step 2: DEGRADED -> NORMAL
    time.sleep(0.03)
    recovered_state2 = cb.fsm.check_recovery()
    assert recovered_state2 == CircuitState.NORMAL
