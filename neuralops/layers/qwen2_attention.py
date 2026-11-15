#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Qwen2 Grouped-Query Attention (GQA) Layer.

Faithfully implements the Hugging Face Qwen2Attention specification:
    - 14:2 or 28:4 GQA head expansion ratio
    - QKV projection with bias
    - Rotary Position Embedding (RoPE)
    - Output projection without bias
"""

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


@ATTENTION_REGISTRY.register("qwen2_attention")
@ATTENTION_REGISTRY.register("qwen_attention", is_alias=True)
class Qwen2Attention(BaseAttention):
    """Qwen2 attention layer with native GQA and QKV bias.

    Args:
        config: Model configuration object.
        hidden_size: Hidden size dimension (default: 896 for Qwen2.5-0.5B).
        num_heads: Number of query heads (default: 14).
        num_kv_heads: Number of key/value heads (default: 2).
        head_dim: Head dimension (default: 64).
    """

    supports_causal: bool = True
    supports_gqa: bool = True

    def __init__(
        self,
        config: Any | None = None,
        hidden_size: int = 896,
        num_heads: int = 14,
        num_kv_heads: int = 2,
        head_dim: int = 64,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, **kwargs)
        self.hidden_size = int(getattr(config, "hidden_size", hidden_size))
        self.num_heads = int(getattr(config, "num_attention_heads", num_heads))
        self.num_kv_heads = int(getattr(config, "num_key_value_heads", num_kv_heads))
        self.head_dim = int(getattr(config, "head_dim", head_dim))

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass matching HF Qwen2Attention return contract.

        Returns:
            Tuple of (output, attn_weights, past_key_value).
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

        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=-2)
            v = torch.cat([past_key_value[1], v], dim=-2)

        present_kv = (k, v) if use_cache else None

        k_rep = repeat_kv(k, self.num_heads // self.num_kv_heads)
        v_rep = repeat_kv(v, self.num_heads // self.num_kv_heads)

        is_causal = kwargs.get("is_causal", kwargs.get("causal", True if L > 1 else False))
        out = F.scaled_dot_product_attention(
            q,
            k_rep,
            v_rep,
            attn_mask=attention_mask,
            is_causal=is_causal if attention_mask is None else False,
        )
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.o_proj(out), None, present_kv


__all__ = ["Qwen2Attention"]
