#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Bidirectional Attention Layer.

Explicit non-causal bidirectional multi-head attention where all sequence positions
freely attend to all other sequence positions (BERT/RoBERTa encoder architecture).
"""

from __future__ import annotations

from typing import Any

from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.layers.full_attention import FullAttention

__all__ = [
    "BidirectionalAttention",
    "bidirectional_attention",
]


@ATTENTION_REGISTRY.register(
    "bidirectional_attention",
    supports_causal=False,
    supports_bidirectional=True,
    supports_gqa=True,
    supports_rope=True,
)
class BidirectionalAttention(FullAttention):
    """Bidirectional (non-causal) multi-head attention layer.

    Attributes:
        supports_causal: False.
        supports_bidirectional: True.
    """

    FAMILY: str = "dense"
    supports_causal: bool = False
    supports_bidirectional: bool = True

    def __init__(
        self,
        config: Any | None = None,  # reason: HuggingFace config or dict
        rank: int | None = None,
        dim: int | None = None,
        num_heads: int | None = None,
        **kwargs: Any,  # reason: arbitrary hyperparameters
    ) -> None:
        """Initializes BidirectionalAttention with is_decoder=False."""
        super().__init__(
            config=config,
            rank=rank,
            dim=dim,
            num_heads=num_heads,
            is_decoder=False,
            **kwargs,
        )
        self.is_decoder = False


bidirectional_attention = BidirectionalAttention
