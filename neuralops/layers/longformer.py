#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Longformer sparse attention combining local sliding window and global tokens."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.base import BaseAttention
from neuralops.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register(
    "longformer",
    supports_causal=True,
    supports_bidirectional=True,
    supports_gqa=True,
)
@ATTENTION_REGISTRY.register(
    "longformer_attention",
    is_alias=True,
    supports_causal=True,
    supports_bidirectional=True,
    supports_gqa=True,
)
class LongformerAttention(BaseAttention):
    """Longformer Attention layer with sliding window and global token attention."""

    supports_causal: bool = True
    supports_bidirectional: bool = True
    supports_gqa: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,
        rank: int | None = None,
        window_size: int = 128,
        num_global_tokens: int = 2,
        dim: int | None = None,
        num_heads: int = 4,
        is_causal: bool = False,
        dropout: float = 0.0,
        **kwargs: Any,
    ) -> None:
        """Initialize LongformerAttention.

        Args:
            config: Optional model config.
            rank: Optional rank parameter.
            window_size: Sliding window radius.
            num_global_tokens: Number of leading tokens with global attention.
            dim: Total embedding dimension.
            num_heads: Head count.
            is_causal: Autoregressive flag.
            dropout: Attention dropout probability.
            **kwargs: Extra hyperparameters.
        """
        super().__init__(config=config if not isinstance(config, int) else None, **kwargs)

        resolved_dim = (
            dim
            or kwargs.get("hidden_size")
            or kwargs.get("d_model")
            or (getattr(config, "hidden_size", getattr(config, "dim", None)) if config is not None else None)
            or 64
        )
        self.dim = int(resolved_dim)
        self.hidden_size = self.dim

        resolved_heads = (
            kwargs.get("num_heads")
            or kwargs.get("num_attention_heads")
            or (
                getattr(config, "num_attention_heads", getattr(config, "num_heads", None))
                if config is not None
                else None
            )
            or num_heads
        )
        self.num_heads = max(1, int(resolved_heads))
        self.head_dim = max(1, self.dim // self.num_heads)

        self.window_size = max(1, int(kwargs.get("window_size", window_size)))
        self.num_global_tokens = max(1, int(kwargs.get("num_global_tokens", num_global_tokens)))
        self.is_causal = bool(kwargs.get("is_causal", is_causal))

        self.q_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.k_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.v_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.out_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.dropout = nn.Dropout(float(dropout))

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        global_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        """Forward pass for LongformerAttention.

        Args:
            x: Input tensor [B, N, D].
            attention_mask: Optional padding mask.
            global_mask: Optional boolean mask [B, N] indicating global tokens.
            **kwargs: Extra parameters.

        Returns:
            Tuple of (output, None).
        """
        B, N, D = x.shape
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        idx = torch.arange(N, device=x.device)
        dist = idx.unsqueeze(1) - idx.unsqueeze(0)

        if self.is_causal:
            local_mask = (dist >= 0) & (dist <= self.window_size)
        else:
            local_mask = dist.abs() <= self.window_size

        if global_mask is not None:
            g_indices = global_mask.bool().unsqueeze(1)
            g_mask = g_indices.unsqueeze(-1) | g_indices.unsqueeze(-2)
        else:
            g_count = min(self.num_global_tokens, N)
            g_mask = (idx.unsqueeze(1) < g_count) | (idx.unsqueeze(0) < g_count)
            g_mask = g_mask.unsqueeze(0).unsqueeze(0)

        full_mask = local_mask.unsqueeze(0).unsqueeze(0) | g_mask
        if self.is_causal:
            full_mask = full_mask & (dist >= 0).unsqueeze(0).unsqueeze(0)

        if attention_mask is not None:
            if attention_mask.ndim == 2:
                full_mask = full_mask & (attention_mask.unsqueeze(1).unsqueeze(2) > 0)
            elif attention_mask.ndim == 4:
                full_mask = full_mask & (attention_mask > 0)

        scores = scores.masked_fill(~full_mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        weights = self.dropout(weights)

        context = torch.matmul(weights, v)
        out = context.transpose(1, 2).contiguous().view(B, N, D)
        out = self.out_proj(out)
        return out, None
