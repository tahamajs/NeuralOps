#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Multi-Head Attention (MHA) Layer.

Canonical multi-head attention module providing classical Vaswani et al. (2017)
multi-head self and cross attention mechanisms with KV caching and GQA expansion.
"""

from __future__ import annotations

from typing import Any

from neuralops.layers.full_attention import FullAttention

__all__ = [
    "MHA",
    "FullAttention",
    "MultiHeadAttention",
    "mha",
]


class MultiHeadAttention(FullAttention):
    """Multi-Head Attention (MHA) neural operator.

    Classical Multi-Head Scaled Dot-Product Attention as introduced in 'Attention Is All You Need'.
    """

    FAMILY: str = "dense"


MHA = MultiHeadAttention
mha = MultiHeadAttention
