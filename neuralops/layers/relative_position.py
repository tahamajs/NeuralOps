#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Relative Position Representation & Bias Layer.

Implements relative position encoding (Shaw et al., 2018; Raffel et al., T5 2020)
using bucketed relative distances added directly to attention logits.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.attention_mask import get_token_valid_mask
from neuralops.core.attention_utils import LoRALinear
from neuralops.core.base import BaseAttention
from neuralops.core.common import _cfg_get
from neuralops.core.registry import ATTENTION_REGISTRY

logger = logging.getLogger(__name__)


def relative_position_bucket(
    relative_position: torch.Tensor,
    bidirectional: bool = True,
    num_buckets: int = 32,
    max_distance: int = 128,
) -> torch.Tensor:
    """Translates relative position distances into discrete bucket indices (T5 algorithm).

    Args:
        relative_position: Tensor of signed or unsigned relative token differences.
        bidirectional: Whether negative differences are mapped to separate buckets.
        num_buckets: Total number of discrete distance buckets.
        max_distance: Maximum distance beyond which logarithmic bucketing applies.

    Returns:
        Tensor of bucket indices of identical shape to relative_position.
    """
    ret = 0
    n = -relative_position
    if bidirectional:
        num_buckets //= 2
        ret = ret + (n < 0).to(torch.long) * num_buckets
        n = torch.abs(n)
    else:
        n = torch.max(n, torch.zeros_like(n))

    max_exact = num_buckets // 2
    is_small = n < max_exact

    val_if_large = max_exact + (
        torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
    ).to(torch.long)
    val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

    return ret + torch.where(is_small, n, val_if_large)


class RelativePositionBias(nn.Module):
    """Computes bucketed relative position bias tables for attention scores."""

    def __init__(
        self,
        num_heads: int,
        num_buckets: int = 32,
        max_distance: int = 128,
        bidirectional: bool = True,
    ) -> None:
        """Initializes RelativePositionBias embedding table."""
        super().__init__()
        self.num_heads: int = num_heads
        self.num_buckets: int = num_buckets
        self.max_distance: int = max_distance
        self.bidirectional: bool = bidirectional
        self.relative_attention_bias = nn.Embedding(self.num_buckets, self.num_heads)

    def forward(self, query_length: int, key_length: int, device: torch.device) -> torch.Tensor:
        """Computes relative position bias of shape (1, num_heads, query_length, key_length)."""
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position
        rp_bucket = relative_position_bucket(
            relative_position,
            bidirectional=self.bidirectional,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
        )
        values = self.relative_attention_bias(rp_bucket)  # (q, k, heads)
        return values.permute([2, 0, 1]).unsqueeze(0)  # (1, heads, q, k)


@ATTENTION_REGISTRY.register(
    "relative_position_attention",
    supports_causal=True,
    supports_bidirectional=True,
    supports_gqa=True,
)
class RelativePositionAttention(BaseAttention):
    """Attention mechanism augmented with learned relative position biases.

    Attributes:
        supports_causal: True.
        supports_bidirectional: True.
        supports_gqa: True.
    """

    FAMILY: str = "modern"
    supports_causal: bool = True
    supports_bidirectional: bool = True
    supports_gqa: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,  # reason: HuggingFace config or dict
        dim: int | None = None,
        num_heads: int | None = None,
        head_dim: int | None = None,
        num_buckets: int = 32,
        max_distance: int = 128,
        dropout: float = 0.0,
        bias: bool = True,
        is_decoder: bool = False,
        **kwargs: Any,  # reason: arbitrary hyperparameters
    ) -> None:
        """Initializes RelativePositionAttention."""
        super().__init__(config=config)

        resolved_dim = (
            dim
            or kwargs.get("hidden_dim")
            or kwargs.get("hidden_size")
            or kwargs.get("d_model")
            or _cfg_get(config, "hidden_dim", None)
            or _cfg_get(config, "hidden_size", None)
            or _cfg_get(config, "dim", 768)
        )
        self.dim: int = int(resolved_dim)
        self.hidden_size: int = self.dim

        resolved_heads = (
            num_heads
            or kwargs.get("num_attention_heads")
            or _cfg_get(config, "num_attention_heads", None)
            or _cfg_get(config, "num_heads", 8)
        )
        self.num_heads: int = max(1, int(resolved_heads))

        resolved_head_dim = head_dim or (self.dim // self.num_heads)
        self.head_dim: int = int(resolved_head_dim)
        self.all_head_size: int = self.num_heads * self.head_dim
        self.is_decoder: bool = bool(is_decoder or _cfg_get(config, "is_decoder", False))

        self.q_proj = LoRALinear(self.dim, self.all_head_size, bias=bias, freeze_base=True)
        self.k_proj = LoRALinear(self.dim, self.all_head_size, bias=bias, freeze_base=True)
        self.v_proj = LoRALinear(self.dim, self.all_head_size, bias=bias, freeze_base=True)
        self.out_proj = LoRALinear(self.all_head_size, self.dim, bias=bias, freeze_base=True)

        self.rel_bias = RelativePositionBias(
            num_heads=self.num_heads,
            num_buckets=num_buckets,
            max_distance=max_distance,
            bidirectional=not self.is_decoder,
        )

        dropout_p = float(dropout or _cfg_get(config, "dropout", 0.0))
        self.dropout = nn.Dropout(dropout_p)
        self.scale: float = 1.0 / math.sqrt(max(self.head_dim, 1))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *args: Any,  # reason: absorbs unused positional args
        **kwargs: Any,  # reason: absorbs unused kwargs
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass with additive bucketed relative position bias."""
        b, seq_len, _ = hidden_states.shape
        h, d = self.num_heads, self.head_dim

        q = self.q_proj(hidden_states).view(b, seq_len, h, d).transpose(1, 2)
        k = self.k_proj(hidden_states).view(b, seq_len, h, d).transpose(1, 2)
        v = self.v_proj(hidden_states).view(b, seq_len, h, d).transpose(1, 2)

        scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * self.scale
        rel_pos_bias = self.rel_bias(seq_len, seq_len, hidden_states.device)
        scores = scores + rel_pos_bias.to(scores.dtype)

        if self.is_decoder and seq_len > 1:
            causal_mask = torch.triu(
                torch.ones((seq_len, seq_len), device=hidden_states.device, dtype=torch.bool), diagonal=1
            )
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), -1e4)

        if attention_mask is not None:
            if attention_mask.dim() == 4:
                scores = scores + attention_mask.to(scores.dtype)
            else:
                valid_m = get_token_valid_mask(attention_mask, b, seq_len, hidden_states.device)
                if valid_m is not None:
                    scores = scores.masked_fill(~valid_m[:, None, None, :], -1e4)

        weights = F.softmax(scores, dim=-1).to(hidden_states.dtype)
        weights = torch.nan_to_num(weights, nan=0.0)
        weights = self.dropout(weights)

        context = torch.matmul(weights, v).transpose(1, 2).contiguous().view(b, seq_len, self.all_head_size)
        out = self.out_proj(context)
        return out, weights


relative_position = RelativePositionAttention

__all__ = [
    "RelativePositionAttention",
    "RelativePositionBias",
    "relative_position",
    "relative_position_bucket",
]
