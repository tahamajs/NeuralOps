#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Linear Attention layer (Decay & Kernel Linear Attention).
"""

from neuralops.layers.gated.decay_linear_attention import DecayLinearAttention
from neuralops.layers.hybrid.positive_kernel_linear_attention import (
    PositiveKernelLinearAttention,
)

LinearAttention = DecayLinearAttention

__all__ = [
    "DecayLinearAttention",
    "LinearAttention",
    "PositiveKernelLinearAttention",
]
