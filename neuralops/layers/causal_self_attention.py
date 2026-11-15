#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Causal Self-Attention Layer.

Enforces strict causal lower-triangular attention masking preventing future-token leakage
during autoregressive decoding, with stateful KV cache threading and rotary embedding hooks.
"""

from __future__ import annotations

from typing import Any

from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.layers.full_attention import FullAttention

__all__ = [
    "CausalSelfAttention",
    "causal_self_attention",
]


@ATTENTION_REGISTRY.register(
    "causal_self_attention",
    supports_causal=True,
    supports_bidirectional=False,
    supports_gqa=True,
    supports_rope=True,
)
@ATTENTION_REGISTRY.register(
    "causal_attention",
    is_alias=True,
    supports_causal=True,
    supports_bidirectional=False,
    supports_gqa=True,
    supports_rope=True,
)
class CausalSelfAttention(FullAttention):
    """Causal self-attention layer strictly enforcing autoregressive triangular masking.

    Attributes:
        supports_causal: True.
        supports_bidirectional: False.
    """

    FAMILY: str = "dense"
    supports_causal: bool = True
    supports_bidirectional: bool = False

    def __init__(
        self,
        config: Any | None = None,  # reason: HuggingFace config or dict
        rank: int | None = None,
        dim: int | None = None,
        num_heads: int | None = None,
        **kwargs: Any,  # reason: arbitrary hyperparameters
    ) -> None:
        """Initializes CausalSelfAttention with is_decoder=True."""
        super().__init__(
            config=config,
            rank=rank,
            dim=dim,
            num_heads=num_heads,
            is_decoder=True,
            **kwargs,
        )
        self.is_decoder = True


causal_self_attention = CausalSelfAttention
