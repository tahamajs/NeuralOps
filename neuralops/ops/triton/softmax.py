#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Softmax, Entmax, and Activation Triton Operators.

Forwarding interface for fused activation and normalization routines.
"""

from __future__ import annotations

from neuralops.ops.triton.elementwise import (
    fast_gelu,
    fused_entmax_bisect_pytorch,
    fused_entmax_bisect_triton,
    softplus_triton,
)
from neuralops.ops.triton.softplus import softplus

__all__ = [
    "softplus",
    "softplus_triton",
    "fused_entmax_bisect_pytorch",
    "fused_entmax_bisect_triton",
    "fast_gelu",
]
