#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Fused Linear Projection + Cross-Entropy Loss Operators.

Forwarding interface for chunked projection cross-entropy preventing logits materialization in HBM.
"""

from __future__ import annotations

from neuralops.ops.triton.fused_ce import (
    FusedLinearCrossEntropy,
    FusedLinearCrossEntropyFunction,
    fused_linear_cross_entropy,
)

__all__ = [
    "FusedLinearCrossEntropy",
    "FusedLinearCrossEntropyFunction",
    "fused_linear_cross_entropy",
]
