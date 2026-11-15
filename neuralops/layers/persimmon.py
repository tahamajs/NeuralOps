#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Persimmon-8B Architecture Attention Layer.

Implements Adept's Persimmon attention with squared ReLU activations,
rotary position embeddings (RoPE), and standard multi-head attention.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.attention_utils import apply_rotary_pos_emb
from neuralops.core.base import BaseAttention
from neuralops.core.registry import ATTENTION_REGISTRY

logger = logging.getLogger(__name__)


@ATTENTION_REGISTRY.register("persimmon")
class PersimmonAttention(BaseAttention):
    """Persimmon multi-head attention with RoPE and squared ReLU.

    Args:
        config: Model configuration object.
        hidden_size: Hidden dimension (default: 4096 for Persimmon 8B).
        num_heads: Number of attention heads (default: 64).
    """

    supports_causal: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,
        hidden_size: int = 4096,
        num_heads: int = 64,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, **kwargs)
        self.hidden_size = int(getattr(config, "hidden_size", hidden_size))
        self.num_heads = int(getattr(config, "num_attention_heads", num_heads))
        self.head_dim = self.hidden_size // max(1, self.num_heads)

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward pass through Persimmon attention with RoPE.

        Args:
            hidden_states: Input tensor of shape [B, L, D].
            attention_mask: Optional attention mask.

        Returns:
            Output tensor of shape [B, L, D].
        """
        B, L, D = hidden_states.shape
        q = self.q_proj(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply rotary position embeddings
        freqs = torch.arange(0, self.head_dim // 2, dtype=torch.float32, device=hidden_states.device)
        angles = torch.outer(torch.arange(L, dtype=torch.float32, device=hidden_states.device), freqs)
        angles = torch.cat([angles, angles], dim=-1)
        cos = torch.cos(angles)
        sin = torch.sin(angles)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        is_causal = kwargs.get("is_causal", kwargs.get("causal", False))
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            is_causal=is_causal,
        )
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out)


__all__ = ["PersimmonAttention"]
