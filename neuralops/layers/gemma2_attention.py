#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Gemma and Gemma-2 Attention Layer with Logit Soft-Capping.

Faithfully implements the Hugging Face Gemma2Attention specification:
    - Head dimension head_dim=256
    - RoPE rotary position embeddings
    - Attention logit soft-capping: tanh(logits / cap) * cap (default cap: 50.0)
    - Grouped-Query Attention (GQA)
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.attention_utils import apply_rotary_pos_emb
from neuralops.core.base import BaseAttention
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.core.utils.gqa import repeat_kv

logger = logging.getLogger(__name__)


@ATTENTION_REGISTRY.register("gemma2_attention")
@ATTENTION_REGISTRY.register("gemma_attention", is_alias=True)
class Gemma2Attention(BaseAttention):
    """Gemma-2 attention layer with logit soft-capping and GQA.

    Args:
        config: Model configuration object.
        hidden_size: Model embedding dimension (default: 2048).
        num_heads: Number of attention query heads (default: 8).
        num_kv_heads: Number of key/value heads (default: 1 for Gemma 2B).
        head_dim: Attention head dimension (default: 256 for Gemma).
        attn_logit_softcapping: Attention logit soft-capping threshold (default: 50.0).
    """

    supports_causal: bool = True
    supports_gqa: bool = True

    def __init__(
        self,
        config: Any | None = None,
        hidden_size: int = 2048,
        num_heads: int = 8,
        num_kv_heads: int = 1,
        head_dim: int = 256,
        attn_logit_softcapping: float = 50.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, **kwargs)
        self.hidden_size = int(getattr(config, "hidden_size", hidden_size))
        self.num_heads = int(getattr(config, "num_attention_heads", num_heads))
        self.num_kv_heads = int(getattr(config, "num_key_value_heads", num_kv_heads))
        self.head_dim = int(getattr(config, "head_dim", head_dim))
        self.soft_capping = float(getattr(config, "attn_logit_softcapping", attn_logit_softcapping))

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass with logit soft-capping.

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

        # Scaled dot-product with soft-capping
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k_rep.transpose(-2, -1)) * scale

        if self.soft_capping > 0.0:
            scores = scores / self.soft_capping
            scores = torch.tanh(scores)
            scores = scores * self.soft_capping

        if attention_mask is not None:
            scores = scores + attention_mask
        elif kwargs.get("is_causal", True) and L > 1:
            causal_mask = torch.triu(torch.full((L, k_rep.shape[-2]), float("-inf"), device=scores.device), diagonal=1)
            scores = scores + causal_mask

        probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(hidden_states.dtype)
        out = torch.matmul(probs, v_rep)
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.o_proj(out), None, present_kv


GemmaAttention = Gemma2Attention

__all__ = ["Gemma2Attention", "GemmaAttention"]
