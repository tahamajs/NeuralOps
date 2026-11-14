#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Fused RMSNorm Operators.

Forwarding interface for fused RMSNorm and fused RMSNorm-linear layers.
"""

from __future__ import annotations

from neuralops.ops.triton.elementwise import (
    fused_rms_variance_match_triton,
    fused_rmsnorm,
    rmsnorm_triton,
)
from neuralops.ops.triton.fused_norm_linear import (
    FusedRMSNormLinear,
    fused_norm_linear,
    fused_norm_linear_pytorch,
    fused_rmsnorm_gate,
    fused_rmsnorm_linear,
)

__all__ = [
    "fused_rmsnorm",
    "rmsnorm_triton",
    "fused_rms_variance_match_triton",
    "FusedRMSNormLinear",
    "fused_norm_linear",
    "fused_norm_linear_pytorch",
    "fused_rmsnorm_linear",
    "fused_rmsnorm_gate",
]
