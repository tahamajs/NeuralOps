#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
L2 Numerical Validation Gates Package.
"""

from __future__ import annotations

from neuralops.validation.gates.numerical.forward_pass import (
    ForwardPassRunsGate,
    _get_dummy_input,
)
from neuralops.validation.gates.numerical.no_nan_inf import NoNanInfGate

__all__ = [
    "ForwardPassRunsGate",
    "NoNanInfGate",
    "_get_dummy_input",
]
