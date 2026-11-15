#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""T5 Relative Position Self and Cross Attention Layer."""

from __future__ import annotations

import logging
import math
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.base import BaseAttention
from neuralops.core.registry import ATTENTION_REGISTRY

logger = logging.getLogger(__name__)


def _relative_position_bucket(
    relative_position: torch.Tensor,
    bidirectional: bool = True,
    num_buckets: int = 32,
    max_distance: int = 128,
) -> torch.Tensor:
    """Computes relative position bucket indices for T5 relative position bias."""
    relative_buckets = 0
    if bidirectional:
        num_buckets //= 2
        relative_buckets = (relative_position > 0).to(torch.long) * num_buckets
        relative_position = torch.abs(relative_position)
    else:
        relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))

    max_exact = num_buckets // 2
    is_small = relative_position < max_exact

    relative_position_if_large = max_exact + (
        torch.log(relative_position.float() / max_exact)
        / math.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).to(torch.long)
    relative_position_if_large = torch.min(
        relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
    )

    return relative_buckets + torch.where(is_small, relative_position, relative_position_if_large)


@ATTENTION_REGISTRY.register("t5_attention")
class T5Attention(BaseAttention):
    """T5 attention layer with logarithmic relative position bias buckets.

    Args:
        config: Model configuration object.
        d_model: Hidden embedding dimension (default: 512 for T5-small, 768 for T5-base).
        num_heads: Number of attention heads (default: 8).
        is_decoder: Whether operating as causal autoregressive decoder.
        relative_attention_num_buckets: Number of relative bias buckets (default: 32).
    """

    supports_causal: bool = True

    def __init__(
        self,
        config: Any | None = None,
        d_model: int = 768,
        num_heads: int = 8,
        is_decoder: bool = False,
        relative_attention_num_buckets: int = 32,
        has_relative_attention_bias: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, **kwargs)
        self.d_model = int(getattr(config, "d_model", getattr(config, "hidden_size", d_model)))
        self.num_heads = int(getattr(config, "num_heads", getattr(config, "num_attention_heads", num_heads)))
        self.d_kv = int(getattr(config, "d_kv", self.d_model // self.num_heads))
        self.is_decoder = bool(getattr(config, "is_decoder", is_decoder))
        self.num_buckets = int(relative_attention_num_buckets)
        self.has_relative_attention_bias = has_relative_attention_bias

        self.q = nn.Linear(self.d_model, self.num_heads * self.d_kv, bias=False)
        self.k = nn.Linear(self.d_model, self.num_heads * self.d_kv, bias=False)
        self.v = nn.Linear(self.d_model, self.num_heads * self.d_kv, bias=False)
        self.o = nn.Linear(self.num_heads * self.d_kv, self.d_model, bias=False)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(self.num_buckets, self.num_heads)

    def compute_bias(self, query_length: int, key_length: int, device: torch.device) -> torch.Tensor:
        """Computes relative position attention bias tensor [1, num_heads, q_len, k_len]."""
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position
        buckets = _relative_position_bucket(
            relative_position,
            bidirectional=not self.is_decoder,
            num_buckets=self.num_buckets,
        )
        values = self.relative_attention_bias(buckets)  # [q_len, k_len, num_heads]
        return values.permute(2, 0, 1).unsqueeze(0)  # [1, num_heads, q_len, k_len]

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_bias: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through T5 attention with relative position bias.

        Returns:
            Tuple of (output, position_bias).
        """
        B, L, _ = hidden_states.shape
        kv = key_value_states if key_value_states is not None else hidden_states
        S = kv.shape[1]

        q = self.q(hidden_states).view(B, L, self.num_heads, self.d_kv).transpose(1, 2)
        k = self.k(kv).view(B, S, self.num_heads, self.d_kv).transpose(1, 2)
        v = self.v(kv).view(B, S, self.num_heads, self.d_kv).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1))

        if position_bias is None and self.has_relative_attention_bias and key_value_states is None:
            position_bias = self.compute_bias(L, S, hidden_states.device)

        if position_bias is not None:
            scores = scores + position_bias

        if attention_mask is not None:
            scores = scores + attention_mask
        elif self.is_decoder and key_value_states is None and L > 1:
            causal_mask = torch.triu(torch.full((L, S), float("-inf"), device=scores.device), diagonal=1)
            scores = scores + causal_mask

        probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(hidden_states.dtype)
        out = torch.matmul(probs, v)
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.o(out), position_bias


__all__ = ["T5Attention"]
