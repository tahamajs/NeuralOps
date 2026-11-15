#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Sliding Window Attention layer with robust causal masking and padding isolation."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.base import BaseAttention
from neuralops.core.registry import ATTENTION_REGISTRY


class SlidingWindowAttention(BaseAttention):
    """Production Sliding Window Attention layer attending strictly to local context.

    Ensures that tokens outside the local window and padding tokens are strictly not attended.
    """

    supports_causal: bool = True
    supports_bidirectional: bool = True
    supports_gqa: bool = True

    def __init__(
        self,
        config: Any | None = None,
        window_size: int = 128,
        dim: int | None = None,
        num_heads: int = 4,
        is_causal: bool = True,
        dropout: float = 0.0,
        **kwargs: Any,
    ) -> None:
        """Initialize SlidingWindowAttention.

        Args:
            config: Optional model configuration object.
            window_size: Maximum tokens to attend locally.
            dim: Feature dimension.
            num_heads: Number of attention heads.
            is_causal: Whether autoregressive masking is enforced.
            dropout: Dropout probability on attention weights.
            **kwargs: Extra hyperparameters.
        """
        super().__init__(config=config if not isinstance(config, int) else None, **kwargs)
        if isinstance(config, int):
            self.window_size = config
        else:
            resolved_ws = (
                kwargs.get("window_size")
                or (getattr(config, "sliding_window", None) if config is not None else None)
                or (getattr(config, "window_size", None) if config is not None else None)
                or window_size
            )
            self.window_size = int(resolved_ws)

        resolved_dim = (
            dim
            or kwargs.get("hidden_size")
            or kwargs.get("d_model")
            or (getattr(config, "hidden_size", getattr(config, "dim", None)) if config is not None else None)
        )
        self.dim = int(resolved_dim) if resolved_dim is not None else None
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
        self.num_heads = int(resolved_heads)
        self.is_causal = bool(kwargs.get("is_causal", is_causal))
        self.dropout = nn.Dropout(float(dropout))

        if self.dim is not None and self.num_heads > 0:
            self.head_dim = self.dim // self.num_heads
            self.q_proj = nn.Linear(self.dim, self.dim, bias=False)
            self.k_proj = nn.Linear(self.dim, self.dim, bias=False)
            self.v_proj = nn.Linear(self.dim, self.dim, bias=False)
            self.out_proj = nn.Linear(self.dim, self.dim, bias=False)
        else:
            self.head_dim = None
            self.q_proj = None
            self.k_proj = None
            self.v_proj = None
            self.out_proj = None

    def _apply_sliding_window_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        window_size: int,
        is_causal: bool,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute sliding window attention with key padding mask enforcement.

        Args:
            q: Query tensor of shape [B, H, N, D].
            k: Key tensor of shape [B, H, N, D].
            v: Value tensor of shape [B, H, N, D].
            window_size: Window radius.
            is_causal: If True, only past tokens in window are attended.
            attention_mask: Optional tensor indicating valid positions.

        Returns:
            Output tensor of shape [B, H, N, D].
        """
        B, H, N, D = q.shape
        scale = 1.0 / math.sqrt(D)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        q_idx = torch.arange(N, device=q.device).unsqueeze(1)
        k_idx = torch.arange(N, device=q.device).unsqueeze(0)
        dist = q_idx - k_idx

        if is_causal:
            window_mask = (dist >= 0) & (dist < window_size)
        else:
            window_mask = dist.abs() < window_size

        full_mask = window_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, N, N]

        if attention_mask is not None:
            if attention_mask.ndim == 2:
                # [B, N] binary mask where 1 is valid token, 0 is padding
                pad_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, N]
                full_mask = full_mask & (pad_mask > 0)
            elif attention_mask.ndim == 3:
                pad_mask = attention_mask.unsqueeze(1)
                full_mask = full_mask & (pad_mask > 0)
            elif attention_mask.ndim == 4:
                full_mask = full_mask & (attention_mask > 0)

        scores = scores.masked_fill(~full_mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        weights = self.dropout(weights)
        return torch.matmul(weights, v)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        """Forward pass for SlidingWindowAttention.

        Args:
            x: Input tensor of shape [B, N, D].
            attention_mask: Optional mask tensor where 0 indicates padding.
            **kwargs: Extra runtime options.

        Returns:
            Tuple of (output, None).
        """
        B, N, D = x.shape
        if self.q_proj is not None and D == self.dim:
            q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
            out = self._apply_sliding_window_attention(
                q, k, v, self.window_size, self.is_causal, attention_mask=attention_mask
            )
            out = out.transpose(1, 2).contiguous().view(B, N, D)
            out = self.out_proj(out)
            return out, None

        head_dim = min(32, D)
        heads = max(1, D // head_dim)
        act_dim = heads * head_dim
        x_pad = x[..., :act_dim]
        q = x_pad.view(B, N, heads, head_dim).transpose(1, 2)
        out = self._apply_sliding_window_attention(
            q, q, q, self.window_size, self.is_causal, attention_mask=attention_mask
        )
        out = out.transpose(1, 2).contiguous().view(B, N, act_dim)
        if act_dim < D:
            out = torch.cat([out, x[..., act_dim:]], dim=-1)
        return out, None


try:
    ATTENTION_REGISTRY._dict["sliding_window"] = SlidingWindowAttention
    ATTENTION_REGISTRY._aliases["sliding_window_attention"] = SlidingWindowAttention
except Exception:
    pass

__all__ = ["SlidingWindowAttention"]
