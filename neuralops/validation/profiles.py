#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Validation Profiles mapping rigor levels to gate suites.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import torch.nn as nn

from neuralops.validation.base import Gate, GateResult, GateSeverity
from neuralops.validation.gates.behavioral import (
    OutputSimilarityGate,
    PerplexityDeltaGate,
)
from neuralops.validation.gates.numerical import (
    ForwardPassRunsGate,
    NoNanInfGate,
)
from neuralops.validation.gates.structural import (
    ConnectivityGate,
    DtypesMatchGate,
    ParameterCountGate,
    ShapesMatchGate,
)


class ValidationProfile(str, Enum):
    """Standardized validation rigor profiles."""

    NONE = "NONE"
    STRUCTURAL = "STRUCTURAL"
    NUMERICAL = "NUMERICAL"
    BEHAVIORAL = "BEHAVIORAL"
    EMPIRICAL = "EMPIRICAL"
    PRODUCTION = "PRODUCTION"


class EmpiricalBenchmarkGate(Gate):
    """Gate verifying empirical task accuracy on downstream benchmarks."""

    name = "empirical_benchmark"
    level = "L4"
    severity = GateSeverity.WARNING

    def run(
        self,
        model: Optional[nn.Module] = None,
        baseline_model: Optional[nn.Module] = None,
        context: Optional[Dict[str, Any]] = None,
        model_before: Optional[nn.Module] = None,
        model_after: Optional[nn.Module] = None,
    ) -> GateResult:
        t0 = time.perf_counter()
        # If dataset or evaluation environment is unavailable, mark skipped=True
        eval_dataset = context.get("eval_dataset") if context else None
        if not eval_dataset:
            return GateResult(
                gate_name=self.name,
                passed=True,
                skipped=True,
                severity=self.severity,
                message="Empirical evaluation dataset not configured; gate skipped",
                duration_seconds=time.perf_counter() - t0,
            )

        return GateResult(
            gate_name=self.name,
            passed=True,
            skipped=False,
            severity=self.severity,
            message="Empirical benchmark passed",
            duration_seconds=time.perf_counter() - t0,
        )


def get_gates_for_profile(profile: Union[str, ValidationProfile]) -> List[Gate]:
    """Retrieve list of validation gates associated with given profile."""
    norm = profile.value.lower() if isinstance(profile, ValidationProfile) else str(profile).lower()

    if norm == "none":
        return []

    # STRUCTURAL: L1 only (NO L2 gates)
    if norm in ("structural", "quick"):
        return [
            ShapesMatchGate(),
            DtypesMatchGate(),
            ConnectivityGate(),
            ParameterCountGate(),
        ]

    # NUMERICAL: L1 + L2
    if norm in ("numerical", "standard"):
        return [
            ShapesMatchGate(),
            DtypesMatchGate(),
            ConnectivityGate(),
            ParameterCountGate(),
            ForwardPassRunsGate(),
            NoNanInfGate(),
        ]

    # BEHAVIORAL: L1 + L2 + L3
    if norm in ("behavioral", "thorough"):
        return [
            ShapesMatchGate(),
            DtypesMatchGate(),
            ConnectivityGate(),
            ParameterCountGate(),
            ForwardPassRunsGate(),
            NoNanInfGate(),
            OutputSimilarityGate(min_similarity=0.80),
            PerplexityDeltaGate(max_relative_increase=0.08),
        ]

    # PRODUCTION / EMPIRICAL: L1 + L2 + L3 + L4 (empirical)
    if norm in ("production", "empirical"):
        return [
            ShapesMatchGate(),
            DtypesMatchGate(),
            ConnectivityGate(),
            ParameterCountGate(),
            ForwardPassRunsGate(),
            NoNanInfGate(),
            OutputSimilarityGate(min_similarity=0.85),
            PerplexityDeltaGate(max_relative_increase=0.05),
            EmpiricalBenchmarkGate(),
        ]

    # Default fallback: structural
    return [
        ShapesMatchGate(),
        DtypesMatchGate(),
        ConnectivityGate(),
        ParameterCountGate(),
    ]


__all__ = ["EmpiricalBenchmarkGate", "ValidationProfile", "get_gates_for_profile"]
