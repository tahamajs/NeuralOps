#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Reformer Locality-Sensitive Hashing attention layer."""

from __future__ import annotations

from typing import Any

from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.layers.lsh_attention import LSHAttention


@ATTENTION_REGISTRY.register(
    "reformer_lsh",
    supports_causal=True,
    supports_bidirectional=True,
    supports_gqa=True,
)
@ATTENTION_REGISTRY.register(
    "reformer_lsh_attention",
    is_alias=True,
    supports_causal=True,
    supports_bidirectional=True,
    supports_gqa=True,
)
class ReformerLSHAttention(LSHAttention):
    """Reformer LSH Attention with multi-round hash bucketing."""

    def __init__(self, config: Any | None = None, **kwargs: Any) -> None:
        super().__init__(config=config, **kwargs)


__all__ = ["ReformerLSHAttention"]
