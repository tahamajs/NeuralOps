#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Mixture of Attentions (MoA) layer and routing engine."""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
import torch.nn as nn

from neuralops.core.base import BaseAttention
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.surgery.moa_router import (
    AttentionEntropyProfile,
    MoARouter,
    compute_attention_entropy,
    profile_layer_sensitivity,
)

logger = logging.getLogger(__name__)


@ATTENTION_REGISTRY.register("mixture_of_attentions")
@ATTENTION_REGISTRY.register("moa", is_alias=True)
class MixtureOfAttentions(BaseAttention):
    """Mixture of Attentions (MoA) block.

    Blends full quadratic attention with fast sub-quadratic attention branches
    using either layer-index partitioning (via MoARouter) or learned dynamic routing.

    Args:
        config: Model configuration object.
        full_attn: Full quadratic attention module (optional).
        linear_attn: Linear or SSM attention module (optional).
        layer_idx: Zero-based depth index of this layer.
    """

    def __init__(
        self,
        config: Any | None = None,
        full_attn: Optional[nn.Module] = None,
        linear_attn: Optional[nn.Module] = None,
        layer_idx: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, **kwargs)
        self.layer_idx = int(layer_idx)

        hidden_size = int(getattr(config, "hidden_size", kwargs.get("hidden_size", 768)))
        num_heads = int(getattr(config, "num_attention_heads", kwargs.get("num_heads", 12)))

        if full_attn is not None:
            self.full_attn = full_attn
        else:
            from neuralops.layers.dense.full_attention import FullAttention

            self.full_attn = FullAttention(config=config, **kwargs)

        if linear_attn is not None:
            self.linear_attn = linear_attn
        else:
            from neuralops.layers.linear.cosformer import CosformerAttention

            self.linear_attn = CosformerAttention(config=config, **kwargs)

        self.router = MoARouter(num_layers=max(self.layer_idx + 1, 12))
        self.gate = nn.Linear(hidden_size, 2, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward pass dispatching or blending between full and linear attention.

        Args:
            hidden_states: Input tensor of shape [B, L, D].
            attention_mask: Optional attention mask.

        Returns:
            Output tensor of shape [B, L, D].
        """
        # If router indicates deterministic partition:
        if self.router.is_full_attention(self.layer_idx):
            out = self.full_attn(hidden_states, attention_mask=attention_mask, **kwargs)
            return out[0] if isinstance(out, tuple) else out

        out = self.linear_attn(hidden_states, attention_mask=attention_mask, **kwargs)
        return out[0] if isinstance(out, tuple) else out


__all__ = [
    "MixtureOfAttentions",
    "MoARouter",
    "compute_attention_entropy",
    "profile_layer_sensitivity",
    "AttentionEntropyProfile",
]
