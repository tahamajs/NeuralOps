#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Attention with Linear Biases (ALiBi) Layer.

Applies static, non-learned negative linear distance biases to query-key attention scores
(Press et al., 2021) to eliminate absolute positional embeddings and support length extrapolation.
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


def get_alibi_slopes(num_heads: int) -> torch.Tensor:
    """Compute ALiBi geometric slopes per head (Press et al. 2021, Eq. 5).

    Args:
        num_heads: Total number of attention heads.

    Returns:
        1D float tensor of positive slopes of shape (num_heads,).
    """
    num_heads = int(num_heads)
    if num_heads <= 0:
        return torch.ones(1, dtype=torch.float32)

    n = 2 ** math.floor(math.log2(num_heads))
    base = torch.tensor(
        [2.0 ** (-8.0 / n * (i + 1)) for i in range(n)],
        dtype=torch.float32,
    )
    if n < num_heads:
        extra = torch.tensor(
            [2.0 ** (-8.0 / (2 * n) * (2 * i + 1)) for i in range(num_heads - n)],
            dtype=torch.float32,
        )
        base = torch.cat([base, extra])
    return base[:num_heads]


def build_alibi_bias(
    num_heads: int,
    seq_len_q: int,
    seq_len_k: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    is_causal: bool = True,
) -> torch.Tensor:
    """Constructs the ALiBi additive attention bias tensor.

    Args:
        num_heads: Number of attention heads.
        seq_len_q: Query sequence length.
        seq_len_k: Key/value sequence length.
        device: Device on which to allocate the tensor.
        dtype: Floating point precision.
        is_causal: Whether causal masking is applied.

    Returns:
        ALiBi bias tensor of shape (1, num_heads, seq_len_q, seq_len_k).
    """
    slopes = get_alibi_slopes(num_heads).to(device=device, dtype=dtype)
    q_pos = torch.arange(seq_len_q, device=device).unsqueeze(1) + (seq_len_k - seq_len_q)
    k_pos = torch.arange(seq_len_k, device=device).unsqueeze(0)
    # Distance: (k - q) <= 0 for past tokens
    distance = k_pos - q_pos
    bias = slopes.view(1, num_heads, 1, 1) * distance.unsqueeze(0).unsqueeze(0).to(dtype)
    if is_causal:
        causal_mask = (k_pos > q_pos).unsqueeze(0).unsqueeze(0)
        bias = bias.masked_fill(causal_mask, -1e4)
    return bias


class ALiBiAttention(BaseAttention):
    """ALiBi Attention layer with geometric per-head relative position slopes.

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
        dropout: float = 0.0,
        bias: bool = True,
        is_decoder: bool = True,
        **kwargs: Any,  # reason: arbitrary hyperparameters
    ) -> None:
        """Initializes ALiBiAttention."""
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
        self.is_decoder: bool = bool(is_decoder if is_decoder is not None else _cfg_get(config, "is_decoder", True))

        self.q_proj = LoRALinear(self.dim, self.all_head_size, bias=bias, freeze_base=True)
        self.k_proj = LoRALinear(self.dim, self.all_head_size, bias=bias, freeze_base=True)
        self.v_proj = LoRALinear(self.dim, self.all_head_size, bias=bias, freeze_base=True)
        self.out_proj = LoRALinear(self.all_head_size, self.dim, bias=bias, freeze_base=True)

        dropout_p = float(dropout or _cfg_get(config, "dropout", 0.0))
        self.dropout = nn.Dropout(dropout_p)
        self.scale: float = 1.0 / math.sqrt(max(self.head_dim, 1))

        # Register non-persistent buffer for slopes
        slopes = get_alibi_slopes(self.num_heads)
        self.register_buffer("slopes", slopes, persistent=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *args: Any,  # reason: absorbs unused positional args
        **kwargs: Any,  # reason: absorbs unused kwargs
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass with ALiBi relative position bias addition."""
        b, seq_len, _ = hidden_states.shape
        h, d = self.num_heads, self.head_dim

        q = self.q_proj(hidden_states).view(b, seq_len, h, d).transpose(1, 2)
        k = self.k_proj(hidden_states).view(b, seq_len, h, d).transpose(1, 2)
        v = self.v_proj(hidden_states).view(b, seq_len, h, d).transpose(1, 2)

        scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * self.scale
        alibi_bias = build_alibi_bias(
            h,
            seq_len,
            seq_len,
            device=hidden_states.device,
            dtype=scores.dtype,
            is_causal=self.is_decoder and seq_len > 1,
        )
        scores = scores + alibi_bias

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
        return out, None


alibi = ALiBiAttention
alibi_attention = ALiBiAttention

__all__ = [
    "ALiBiAttention",
    "alibi",
    "alibi_attention",
    "build_alibi_bias",
    "get_alibi_slopes",
]
