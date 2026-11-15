#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Phi-3.5 MoE Architecture Attention Layer.

Combines standard dense multi-head attention with sparse mixture of experts (MoE)
routing and load-balancing auxiliary loss computation.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
import torch.nn as nn

from neuralops.core.base import BaseAttention
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.layers.dense.full_attention import FullAttention
from neuralops.layers.sparse.reap_moe import REAPMoE

logger = logging.getLogger(__name__)


@ATTENTION_REGISTRY.register("phimoe")
class PhiMoEAttention(BaseAttention):
    """PhiMoE block combining dense attention with sparse MoE expert routing.

    Args:
        config: Model configuration object.
        hidden_size: Hidden representation size.
        num_heads: Number of attention heads.
        num_experts: Total number of experts (default: 16).
        top_k: Number of routed experts per token (default: 2).
    """

    supports_causal: bool = True
    supports_gqa: bool = True

    def __init__(
        self,
        config: Any | None = None,
        hidden_size: int = 768,
        num_heads: int = 12,
        num_experts: int = 16,
        top_k: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, **kwargs)
        self.hidden_size = int(getattr(config, "hidden_size", hidden_size))
        self.num_heads = int(getattr(config, "num_attention_heads", num_heads))
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)

        self.attn_norm = nn.LayerNorm(self.hidden_size)
        self.attn = FullAttention(config=config, hidden_size=self.hidden_size, num_heads=self.num_heads, **kwargs)

        self.moe_norm = nn.LayerNorm(self.hidden_size)
        self.moe = REAPMoE(d_model=self.hidden_size, n_experts=self.num_experts, top_k=self.top_k, **kwargs)

    @property
    def aux_loss(self) -> torch.Tensor:
        return self.moe.aux_loss

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward pass through PhiMoE attention and sparse MoE.

        Args:
            hidden_states: Input tensor of shape [B, L, D].
            attention_mask: Optional attention mask.

        Returns:
            Output tensor of shape [B, L, D].
        """
        # Attention pass with residual
        normed_attn = self.attn_norm(hidden_states)
        attn_out = self.attn(normed_attn, attention_mask=attention_mask, **kwargs)
        a_tensor = attn_out[0] if isinstance(attn_out, tuple) else attn_out
        x = hidden_states + a_tensor

        # MoE pass with residual
        moe_out, _ = self.moe(self.moe_norm(x))
        return x + moe_out


__all__ = ["PhiMoEAttention"]
