#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Multi-Head Attention (multi_head_attention) Layer.

Canonical name module for MultiHeadAttention, providing standard scaled dot-product
attention across independent projection subspaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from neuralops.layers.mha import MHA, MultiHeadAttention, mha

__all__ = [
    "MHA",
    "MultiHeadAttention",
    "MultiHeadAttentionConfig",
    "mha",
]


@dataclass(frozen=True)
class MultiHeadAttentionConfig:
    """Configuration schema for MultiHeadAttention.

    Attributes:
        hidden_size: Total embedding dimension.
        num_attention_heads: Number of attention heads.
        dropout_prob: Attention probabilities dropout rate.
        is_decoder: Whether causal masking is enforced.
        bias: Whether linear projection layers include bias parameters.
    """

    hidden_size: int = 768
    num_attention_heads: int = 12
    dropout_prob: float = 0.0
    is_decoder: bool = False
    bias: bool = True
