#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Validation Gates Catalog.
"""

from neuralops.validation.gates.behavioral import (
    AttentionPatternsGate,
    LogitKLGate,
    OutputSimilarityGate,
    PerplexityDeltaGate,
    PerplexityGate,
)
from neuralops.validation.gates.empirical import (
    AccuracyGate,
    LatencyGate,
    NeedleInAHaystackGate,
    ThroughputGate,
    VRAMGate,
)
from neuralops.validation.gates.numerical import (
    ForwardPassRunsGate,
    NoNanInfGate,
)
from neuralops.validation.gates.structural import (
    ConnectivityGate,
    DtypesMatchGate,
    NoOrphanModulesGate,
    ParameterCountGate,
    ShapesMatchGate,
)

__all__ = [
    "AccuracyGate",
    "AttentionPatternsGate",
    "ConnectivityGate",
    "DtypesMatchGate",
    "ForwardPassRunsGate",
    "LatencyGate",
    "LogitKLGate",
    "NeedleInAHaystackGate",
    "NoNanInfGate",
    "NoOrphanModulesGate",
    "OutputSimilarityGate",
    "ParameterCountGate",
    "PerplexityDeltaGate",
    "PerplexityGate",
    "ShapesMatchGate",
    "ThroughputGate",
    "VRAMGate",
]
