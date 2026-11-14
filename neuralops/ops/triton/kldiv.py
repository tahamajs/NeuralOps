#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Fused KL Divergence Loss Operators.

Forwarding interface for temperature-scaled Kullback-Leibler divergence kernels.
"""

from __future__ import annotations

from neuralops.ops.triton.fused_kd import (
    FusedKDLoss,
    FusedLinearKLDivergence,
    FusedLinearKLDivergenceFunction,
    fused_kd_loss,
    fused_kd_loss_pytorch,
    fused_linear_kl_divergence,
)

__all__ = [
    "FusedKDLoss",
    "fused_kd_loss",
    "fused_kd_loss_pytorch",
    "FusedLinearKLDivergence",
    "FusedLinearKLDivergenceFunction",
    "fused_linear_kl_divergence",
]
