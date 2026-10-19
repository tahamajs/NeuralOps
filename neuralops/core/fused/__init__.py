#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
NeuralOps Fused Kernels Core Modules.
"""

from .fused_ce import fused_linear_cross_entropy
from .fused_kd import fused_linear_kl_divergence
from .fused_norm_linear import (
    RMSNormLinear,
    LayerNormLinear,
    RMSNormGated,
    LayerNormGated,
)

__all__ = [
    "fused_linear_cross_entropy",
    "fused_linear_kl_divergence",
    "RMSNormLinear",
    "LayerNormLinear",
    "RMSNormGated",
    "LayerNormGated",
]
