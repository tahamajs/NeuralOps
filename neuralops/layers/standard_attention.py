#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Standard Attention Layer.

Canonical standard attention module wrapping FullAttention with comprehensive parameter
aliasing and compatibility with HuggingFace Transformer backbones.
"""

from __future__ import annotations

from typing import Any

from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.layers.full_attention import FullAttention

__all__ = [
    "StandardAttention",
    "standard_attention",
]


@ATTENTION_REGISTRY.register(
    "standard_attention",
    supports_causal=True,
    supports_bidirectional=True,
    supports_gqa=True,
    supports_rope=True,
)
class StandardAttention(FullAttention):
    """Standard multi-head attention neural operator."""

    FAMILY: str = "dense"


standard_attention = StandardAttention
