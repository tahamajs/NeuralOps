#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Validation Runner executing gate suites across models.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import torch.nn as nn

from neuralops.validation.base import Gate, GateResult, GateSeverity
from neuralops.validation.profiles import get_gates_for_profile
from neuralops.validation.result import ValidationResult
from neuralops.validator.errors import RuntimeValidationError

logger = logging.getLogger("NeuralOps.Validation.Runner")


def run_validation_gates(
    model: nn.Module,
    baseline_model: Optional[nn.Module] = None,
    profile: str = "standard",
    gates: Optional[List[Gate]] = None,
    context: Optional[Dict[str, Any]] = None,
    fail_fast: bool = False,
) -> ValidationResult:
    """Run validation gates against model."""
    t0 = time.perf_counter()
    active_gates = gates if gates is not None else get_gates_for_profile(profile)

    results: List[GateResult] = []
    for gate in active_gates:
        try:
            res = gate.run(model=model, baseline_model=baseline_model, context=context)
        except Exception as e:
            logger.error("Gate '%s' execution raised unexpected error: %s", gate.name, e)
            res = GateResult(
                gate_name=gate.name,
                passed=False,
                severity=gate.severity,
                message=f"Gate raised unhandled exception: {e}",
            )
        results.append(res)

        if not res.passed and res.severity in (GateSeverity.FATAL, GateSeverity.ERROR):
            logger.warning("Validation gate '%s' FAILED: %s", gate.name, res.message)
            if fail_fast:
                raise RuntimeValidationError(
                    f"Validation gate '{gate.name}' failed: {res.message}",
                    error_code="E_GATE_FATAL" if res.severity == GateSeverity.FATAL else "E_GATE_ERROR",
                )

    duration = time.perf_counter() - t0
    return ValidationResult.from_results(profile=profile, results=results, duration_seconds=duration)


__all__ = ["run_validation_gates"]
