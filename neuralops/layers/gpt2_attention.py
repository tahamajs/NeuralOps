#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""GPT-2 Canonical Multi-Head Self-Attention Layer."""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.base import BaseAttention
from neuralops.core.registry import ATTENTION_REGISTRY

logger = logging.getLogger(__name__)


@ATTENTION_REGISTRY.register("gpt2_attention")
class GPT2Attention(BaseAttention):
    """GPT-2 attention layer with combined QKV projection.

    Args:
        config: Model configuration object.
        hidden_size: Embedding dimension (default: 768).
        num_heads: Number of attention heads (default: 12).
    """

    supports_causal: bool = True

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
        self.head_dim = self.hidden_size // self.num_heads

        self.c_attn = nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=True)
        self.c_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass through GPT-2 self-attention.

        Returns:
            Tuple of (output, attn_weights, past_key_value).
        """
        B, L, _ = hidden_states.shape

        qkv = self.c_attn(hidden_states)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=-2)
            v = torch.cat([past_key_value[1], v], dim=-2)

        present_kv = (k, v) if use_cache else None

        is_causal = kwargs.get("is_causal", kwargs.get("causal", True if L > 1 else False))
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            is_causal=is_causal if attention_mask is None else False,
        )
        out = out.transpose(1, 2).contiguous().view(B, L, self.hidden_size)
        return self.c_proj(out), None, present_kv


__all__ = ["GPT2Attention"]
