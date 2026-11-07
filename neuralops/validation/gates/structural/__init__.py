#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
L1 Structural Validation Gates Package.
"""

from __future__ import annotations

from neuralops.validation.gates.structural.connectivity import ConnectivityGate, NoOrphanModulesGate
from neuralops.validation.gates.structural.dtypes import DtypesMatchGate
from neuralops.validation.gates.structural.parameter_count import ParameterCountGate
from neuralops.validation.gates.structural.shapes import ShapesMatchGate

__all__ = [
    "ConnectivityGate",
    "DtypesMatchGate",
    "NoOrphanModulesGate",
    "ParameterCountGate",
    "ShapesMatchGate",
]
