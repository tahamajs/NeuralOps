#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Falcon Multi-Query and Multi-Group Attention Layer."""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.attention_utils import apply_rotary_pos_emb
from neuralops.core.base import BaseAttention
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.core.utils.gqa import repeat_kv

logger = logging.getLogger(__name__)


@ATTENTION_REGISTRY.register("falcon_attention")
class FalconAttention(BaseAttention):
    """Falcon multi-query / multi-group attention.

    Args:
        config: Model configuration object.
        hidden_size: Hidden dimension (default: 4544 for Falcon-7B).
        num_heads: Number of attention heads (default: 71).
        num_kv_heads: Number of key-value heads (default: 1 for Falcon MQA).
    """

    supports_causal: bool = True
    supports_gqa: bool = True

    def __init__(
        self,
        config: Any | None = None,
        hidden_size: int = 4544,
        num_heads: int = 71,
        num_kv_heads: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, **kwargs)
        self.hidden_size = int(getattr(config, "hidden_size", hidden_size))
        self.num_heads = int(getattr(config, "num_attention_heads", num_heads))
        self.num_kv_heads = int(getattr(config, "num_kv_heads", num_kv_heads))
        self.head_dim = self.hidden_size // self.num_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.dense = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward pass through Falcon attention.

        Args:
            hidden_states: Tensor [B, L, D].
            attention_mask: Optional attention mask.

        Returns:
            Output tensor [B, L, D].
        """
        B, L, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)

        freqs = torch.arange(0, self.head_dim // 2, dtype=torch.float32, device=hidden_states.device)
        angles = torch.outer(torch.arange(L, dtype=torch.float32, device=hidden_states.device), freqs)
        angles = torch.cat([angles, angles], dim=-1)
        cos = torch.cos(angles)
        sin = torch.sin(angles)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        k_rep = repeat_kv(k, self.num_heads // self.num_kv_heads)
        v_rep = repeat_kv(v, self.num_heads // self.num_kv_heads)

        is_causal = kwargs.get("is_causal", kwargs.get("causal", True if L > 1 else False))
        out = F.scaled_dot_product_attention(q, k_rep, v_rep, attn_mask=attention_mask, is_causal=is_causal)
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.dense(out)


__all__ = ["FalconAttention"]
