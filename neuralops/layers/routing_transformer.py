#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Routing Transformer clustering attention layer."""

from __future__ import annotations

from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.layers.sparse.routing_attention import (
    RoutingAttention as _BaseRouting,
    inject_routing,
)


@ATTENTION_REGISTRY.register(
    "routing_transformer",
    supports_causal=True,
    supports_bidirectional=True,
    supports_gqa=True,
)
@ATTENTION_REGISTRY.register(
    "routing_transformer_attention",
    is_alias=True,
    supports_causal=True,
    supports_bidirectional=True,
    supports_gqa=True,
)
class RoutingTransformerAttention(_BaseRouting):
    """Routing Transformer spherical k-means clustering attention."""


RoutingAttention = _BaseRouting
__all__ = ["RoutingTransformerAttention", "RoutingAttention", "inject_routing"]
