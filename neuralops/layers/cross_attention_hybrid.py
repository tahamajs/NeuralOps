#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Cross Attention Hybrid Layer.

Blends self-attention and cross-attention branches via learned gating
for encoder-decoder, multimodal, or retrieval-augmented representations.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.base import BaseAttention
from neuralops.core.registry import ATTENTION_REGISTRY

logger = logging.getLogger(__name__)


@ATTENTION_REGISTRY.register("cross_attention_hybrid")
class CrossAttentionHybrid(BaseAttention):
    """Hybrid layer combining self-attention and cross-attention.

    Args:
        config: Model configuration object.
        hidden_size: Hidden representation size.
        num_heads: Number of attention heads.
    """

    supports_causal: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,
        hidden_size: int = 768,
        num_heads: int = 12,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, **kwargs)
        self.hidden_size = int(getattr(config, "hidden_size", hidden_size))
        self.num_heads = int(getattr(config, "num_attention_heads", num_heads))
        self.head_dim = self.hidden_size // max(1, self.num_heads)

        # Self-attention projections
        self.q_self = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_self = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_self = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.out_self = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        # Cross-attention projections
        self.q_cross = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_cross = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_cross = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.out_cross = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        # Learnable convex combination gate
        self.gate = nn.Parameter(torch.zeros(self.hidden_size))

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward pass computing gated self- and cross-attention.

        Args:
            hidden_states: Query input tensor [B, L, D].
            encoder_hidden_states: Optional key/value context tensor [B, S, D].
            attention_mask: Mask for self-attention.
            encoder_attention_mask: Mask for cross-attention.

        Returns:
            Blended output tensor [B, L, D].
        """
        B, L, D = hidden_states.shape
        # 1. Self-attention
        qs = self.q_self(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        ks = self.k_self(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        vs = self.v_self(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        is_causal = kwargs.get("is_causal", kwargs.get("causal", False))
        self_out = F.scaled_dot_product_attention(qs, ks, vs, attn_mask=attention_mask, is_causal=is_causal)
        self_out = self.out_self(self_out.transpose(1, 2).contiguous().view(B, L, D))

        if encoder_hidden_states is None:
            return self_out

        # 2. Cross-attention
        S = encoder_hidden_states.shape[1]
        qc = self.q_cross(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        kc = self.k_cross(encoder_hidden_states).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        vc = self.v_cross(encoder_hidden_states).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        cross_out = F.scaled_dot_product_attention(qc, kc, vc, attn_mask=encoder_attention_mask, is_causal=False)
        cross_out = self.out_cross(cross_out.transpose(1, 2).contiguous().view(B, L, D))

        # 3. Gated blend
        alpha = torch.sigmoid(self.gate)
        return alpha * self_out + (1.0 - alpha) * cross_out


__all__ = ["CrossAttentionHybrid"]
