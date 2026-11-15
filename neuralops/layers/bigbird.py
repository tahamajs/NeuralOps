#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""BigBird Sparse Attention layer combining local window, global tokens, and random blocks."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.base import BaseAttention
from neuralops.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register(
    "bigbird",
    supports_causal=True,
    supports_bidirectional=True,
    supports_gqa=True,
)
@ATTENTION_REGISTRY.register(
    "bigbird_attention",
    is_alias=True,
    supports_causal=True,
    supports_bidirectional=True,
    supports_gqa=True,
)
class BigBirdAttention(BaseAttention):
    """BigBird Sparse Attention layer.

    Unifies three sparsity mechanisms:
    1. Local sliding window attention.
    2. Global tokens attending to and from all positions.
    3. Random block connections for theoretical graph connectivity.
    """

    supports_causal: bool = True
    supports_bidirectional: bool = True
    supports_gqa: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,
        rank: int | None = None,
        window_size: int = 16,
        num_global_tokens: int = 4,
        num_random_blocks: int = 2,
        dim: int | None = None,
        num_heads: int = 4,
        is_causal: bool = False,
        dropout: float = 0.0,
        **kwargs: Any,
    ) -> None:
        """Initialize BigBirdAttention.

        Args:
            config: Optional configuration object.
            rank: Optional rank.
            window_size: Radius of local window.
            num_global_tokens: Number of leading global tokens.
            num_random_blocks: Density factor for random connections.
            dim: Embedding dimension.
            num_heads: Number of attention heads.
            is_causal: If True, enforce causal triangular masking.
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

        self.window_size = int(kwargs.get("window_size", window_size))
        self.num_global_tokens = int(kwargs.get("num_global_tokens", num_global_tokens))
        self.num_random_blocks = int(kwargs.get("num_random_blocks", num_random_blocks))
        self.is_causal = bool(kwargs.get("is_causal", is_causal))

        self.q_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.k_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.v_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.out_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.dropout = nn.Dropout(float(dropout))

    def get_attention_pattern(
        self,
        seq_len: int,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the component masks and the combined BigBird mask.

        Args:
            seq_len: Sequence length N.
            device: Target torch device.

        Returns:
            Tuple of (local_mask, global_mask, random_mask, combined_mask)
            Each tensor is boolean of shape [seq_len, seq_len].
        """
        idx = torch.arange(seq_len, device=device)
        dist = idx.unsqueeze(1) - idx.unsqueeze(0)

        if self.is_causal:
            local_mask = (dist >= 0) & (dist <= self.window_size)
        else:
            local_mask = dist.abs() <= self.window_size

        g_count = min(self.num_global_tokens, seq_len)
        global_mask = (idx.unsqueeze(1) < g_count) | (idx.unsqueeze(0) < g_count)
        if self.is_causal:
            global_mask = global_mask & (dist >= 0)

        # Deterministic pseudo-random pattern based on coprimes
        mod_factor = max(1, seq_len // max(1, self.num_random_blocks))
        random_mask = ((idx.unsqueeze(1) * 37 + idx.unsqueeze(0) * 17) % mod_factor) == 0
        if self.is_causal:
            random_mask = random_mask & (dist >= 0)

        combined_mask = local_mask | global_mask | random_mask
        return local_mask, global_mask, random_mask, combined_mask

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        """Forward pass for BigBirdAttention.

        Args:
            x: Input tensor [B, N, D].
            attention_mask: Optional padding mask.
            **kwargs: Extra runtime options.

        Returns:
            Tuple of (output, None).
        """
        B, N, D = x.shape
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        _, _, _, combined_mask = self.get_attention_pattern(N, device=x.device)
        mask = combined_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, N, N]

        if attention_mask is not None:
            if attention_mask.ndim == 2:
                mask = mask & (attention_mask.unsqueeze(1).unsqueeze(2) > 0)
            elif attention_mask.ndim == 4:
                mask = mask & (attention_mask > 0)

        scores = scores.masked_fill(~mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        weights = self.dropout(weights)

        context = torch.matmul(weights, v)
        out = context.transpose(1, 2).contiguous().view(B, N, D)
        out = self.out_proj(out)
        return out, None
