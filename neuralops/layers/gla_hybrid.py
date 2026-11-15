#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""GLA (Gated Linear Attention) Hybrid Attention Layer Facade."""

from __future__ import annotations

from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.layers.hybrid.halo_regla import HALOReGLAAttention, HaloReGLAAttention
from neuralops.layers.hybrid.regla_attention import ReGLAAttention

GLAHybridAttention = HALOReGLAAttention
ATTENTION_REGISTRY.register("gla_hybrid", is_alias=True)(GLAHybridAttention)

__all__ = [
    "GLAHybridAttention",
    "HALOReGLAAttention",
    "HaloReGLAAttention",
    "ReGLAAttention",
]
