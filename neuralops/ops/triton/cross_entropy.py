#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Fused Cross-Entropy Loss Operators.

Forwarding interface for online softmax fused cross-entropy and projection loss.
"""

from __future__ import annotations

from neuralops.ops.triton.fused_ce import (
    FusedCrossEntropyLoss,
    FusedLinearCrossEntropy,
    FusedLinearCrossEntropyFunction,
    fused_cross_entropy,
    fused_cross_entropy_pytorch,
    fused_linear_cross_entropy,
)

__all__ = [
    "FusedCrossEntropyLoss",
    "fused_cross_entropy",
    "fused_cross_entropy_pytorch",
    "FusedLinearCrossEntropy",
    "FusedLinearCrossEntropyFunction",
    "fused_linear_cross_entropy",
]
