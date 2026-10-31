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

from typing import Dict, List

from neuralops.validation.base import Gate
from neuralops.validation.gates.behavioral import (
    OutputSimilarityGate,
    PerplexityDeltaGate,
)
from neuralops.validation.gates.numerical import (
    ForwardPassRunsGate,
    NoNanInfGate,
)
from neuralops.validation.gates.structural import (
    DtypesMatchGate,
    NoOrphanModulesGate,
    ParameterCountGate,
    ShapesMatchGate,
)


def get_gates_for_profile(profile: str) -> List[Gate]:
    """Retrieve list of validation gates associated with given profile."""
    norm = profile.lower()

    if norm == "none":
        return []

    if norm in ("quick", "structural"):
        return [
            ShapesMatchGate(),
            DtypesMatchGate(),
            ForwardPassRunsGate(),
            ParameterCountGate(),
        ]

    if norm in ("standard", "numerical"):
        return [
            ShapesMatchGate(),
            DtypesMatchGate(),
            NoOrphanModulesGate(),
            ForwardPassRunsGate(),
            NoNanInfGate(),
            ParameterCountGate(),
            OutputSimilarityGate(min_similarity=0.80),
        ]

    if norm in ("thorough", "behavioral", "empirical", "production"):
        return [
            ShapesMatchGate(),
            DtypesMatchGate(),
            NoOrphanModulesGate(),
            ForwardPassRunsGate(),
            NoNanInfGate(),
            ParameterCountGate(),
            OutputSimilarityGate(min_similarity=0.85),
            PerplexityDeltaGate(max_relative_increase=0.08),
        ]

    # Default fallback: standard
    return [
        ShapesMatchGate(),
        DtypesMatchGate(),
        ForwardPassRunsGate(),
        NoNanInfGate(),
        ParameterCountGate(),
    ]


__all__ = ["get_gates_for_profile"]
