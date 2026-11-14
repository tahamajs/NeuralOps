#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Fused Rotary Position Embedding (RoPE) Operators.

Forwarding interface for fused RoPE kernels and analytical autograd passes.
"""

from __future__ import annotations

from neuralops.ops.triton.fused_rope import (
    apply_rope_rotation,
    fused_rope_embedding_kernel,
    fused_unified_rope_offset_triton,
)

__all__ = [
    "fused_rope_embedding_kernel",
    "fused_unified_rope_offset_triton",
    "apply_rope_rotation",
]
