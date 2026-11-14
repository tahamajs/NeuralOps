#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Fused LayerNorm Operators.

Forwarding interface for fused LayerNorm kernel implementations.
"""

from __future__ import annotations

from neuralops.ops.triton.elementwise import layernorm_triton

__all__ = [
    "layernorm_triton",
]
