#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Autonomous Circuit Breaker & Chaos Recovery FSM (Phases 031-034, 086-089).

Implements a high-frequency three-state Finite State Machine:
  Normal (Level 1) -> Degraded (Level 2) -> Tripped (Level 3)
with automatic 20ms cooldown recovery and zero memory leakage across stress iterations.
"""

from __future__ import annotations

import logging
import time
from enum import Enum
from typing import Any, Callable, Dict, Optional

import torch

logger = logging.getLogger(__name__)

__all__ = [
    "CircuitState",
    "CircuitBreakerFSM",
    "AutonomousCircuitBreaker",
]


class CircuitState(str, Enum):
    """Circuit Breaker operational states."""

    NORMAL = "Normal"  # Level 1 active (CUDA C++ / Peak Performance)
    DEGRADED = "Degraded"  # Level 2 active (Triton JIT / Dynamic Tiling)
    TRIPPED = "Tripped"  # Level 3 active (Safe Vectorized PyTorch Reference)


class CircuitBreakerFSM:
    """
    Three-state Circuit Breaker FSM managing fallback transitions and 20ms auto-recovery.
    """

    def __init__(
        self,
        cooldown_seconds: float = 0.020,  # 20ms auto-recovery
        failure_threshold: int = 1,
    ):
        self.cooldown_seconds = cooldown_seconds
        self.failure_threshold = failure_threshold
        self.state = CircuitState.NORMAL
        self.failure_count = 0
        self.last_state_change = time.perf_counter()
        self.total_trips = 0
        self.recovered_ooms = 0

    def record_failure(self, reason: str = "") -> CircuitState:
        """Transitions state downwards upon kernel panic or failure."""
        self.failure_count += 1
        now = time.perf_counter()

        if self.failure_count >= self.failure_threshold:
            if self.state == CircuitState.NORMAL:
                self.state = CircuitState.DEGRADED
                self.failure_count = 0
                self.last_state_change = now
                logger.warning("Circuit Breaker degraded to Level 2 (Triton). Reason: %s", reason)
            elif self.state == CircuitState.DEGRADED:
                self.state = CircuitState.TRIPPED
                self.failure_count = 0
                self.total_trips += 1
                self.last_state_change = now
                logger.error("Circuit Breaker TRIPPED to Level 3 (PyTorch Reference). Reason: %s", reason)

        return self.state

    def check_recovery(self) -> CircuitState:
        """Evaluates whether cooldown period has elapsed to auto-recover."""
        if self.state == CircuitState.NORMAL:
            return self.state

        now = time.perf_counter()
        elapsed = now - self.last_state_change
        if self.cooldown_seconds <= 0:
            levels_to_recover = 2
        else:
            levels_to_recover = int(elapsed / self.cooldown_seconds)

        if levels_to_recover <= 0:
            return self.state

        if self.state == CircuitState.TRIPPED:
            if levels_to_recover >= 2:
                self.state = CircuitState.NORMAL
                self.failure_count = 0
                self.last_state_change = now
                logger.info("Circuit Breaker auto-recovering: TRIPPED -> NORMAL (after %.1fms)", elapsed * 1000)
            elif levels_to_recover == 1:
                self.state = CircuitState.DEGRADED
                self.failure_count = 0
                self.last_state_change = now
                logger.info("Circuit Breaker auto-recovering: TRIPPED -> DEGRADED (after %.1fms)", elapsed * 1000)
        elif self.state == CircuitState.DEGRADED:
            self.state = CircuitState.NORMAL
            self.failure_count = 0
            self.last_state_change = now
            logger.info("Circuit Breaker auto-recovering: DEGRADED -> NORMAL (after %.1fms)", elapsed * 1000)

        return self.state

    def force_reset(self) -> None:
        """Resets the circuit breaker state immediately to NORMAL."""
        self.state = CircuitState.NORMAL
        self.failure_count = 0
        self.last_state_change = time.perf_counter()


class AutonomousCircuitBreaker:
    """
    Autonomous runtime wrapper executing functions across 3 fallback tiers
    under synthetic stress and OOM chaos conditions.
    """

    def __init__(
        self,
        cooldown_seconds: float = 0.020,
        enable_oom_healing: bool = True,
    ):
        self.fsm = CircuitBreakerFSM(cooldown_seconds=cooldown_seconds)
        self.enable_oom_healing = enable_oom_healing
        self.level1_invocations = 0
        self.level2_invocations = 0
        self.level3_invocations = 0

    def execute(
        self,
        level1_fn: Callable[..., torch.Tensor],
        level2_fn: Callable[..., torch.Tensor],
        level3_fn: Callable[..., torch.Tensor],
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Executes with automatic tier fallback and OOM interception.
        """
        # Periodic auto-recovery check
        self.fsm.check_recovery()

        errors: list[tuple[int, Exception]] = []

        # Tier 1: Normal execution
        if self.fsm.state == CircuitState.NORMAL:
            try:
                self.level1_invocations += 1
                return level1_fn(*args, **kwargs)
            except torch.cuda.OutOfMemoryError as oom:
                if self.enable_oom_healing and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                errors.append((1, oom))
                self.fsm.record_failure(f"Level 1 CUDA OOM: {oom}")
            except Exception as exc:
                errors.append((1, exc))
                self.fsm.record_failure(f"Level 1 Kernel Panic: {exc}")

        # Tier 2: Degraded execution
        if self.fsm.state == CircuitState.DEGRADED:
            try:
                self.level2_invocations += 1
                return level2_fn(*args, **kwargs)
            except torch.cuda.OutOfMemoryError as oom:
                if self.enable_oom_healing and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                errors.append((2, oom))
                self.fsm.record_failure(f"Level 2 Triton OOM: {oom}")
            except Exception as exc:
                errors.append((2, exc))
                self.fsm.record_failure(f"Level 2 Triton Exception: {exc}")

        # Tier 3: Tripped execution (Safe PyTorch Vectorized Reference)
        self.level3_invocations += 1
        try:
            return level3_fn(*args, **kwargs)
        except Exception as level3_exc:
            errors.append((3, level3_exc))
            err_details = ", ".join(f"Level {lvl}: {type(err).__name__}: {err}" for lvl, err in errors)
            raise RuntimeError(f"All circuit-breaker levels failed. Errors: {err_details}") from level3_exc
