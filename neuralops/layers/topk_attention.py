#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Top-K Dynamic Sparse Attention layer."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.base import BaseAttention
from neuralops.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register(
    "topk_attention",
    supports_causal=True,
    supports_bidirectional=True,
    supports_gqa=True,
)
@ATTENTION_REGISTRY.register(
    "topk",
    is_alias=True,
    supports_causal=True,
    supports_bidirectional=True,
    supports_gqa=True,
)
class TopKAttention(BaseAttention):
    """Top-K Sparse Attention keeping only the highest-scoring key tokens per query."""

    supports_causal: bool = True
    supports_bidirectional: bool = True
    supports_gqa: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,
        top_k: int = 16,
        dim: int | None = None,
        num_heads: int = 4,
        is_causal: bool = False,
        dropout: float = 0.0,
        **kwargs: Any,
    ) -> None:
        """Initialize TopKAttention.

        Args:
            config: Optional config.
            top_k: Number of key tokens to retain per query.
            dim: Embedding dimension.
            num_heads: Number of attention heads.
            is_causal: Autoregressive flag.
            dropout: Attention dropout.
            **kwargs: Extra parameters.
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

        self.top_k = max(1, int(kwargs.get("top_k", top_k)))
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
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        """Forward pass for TopKAttention.

        Args:
            x: Input tensor [B, N, D].
            attention_mask: Optional padding mask.
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

        if self.is_causal:
            idx = torch.arange(N, device=x.device)
            c_mask = idx.unsqueeze(1) >= idx.unsqueeze(0)
            scores = scores.masked_fill(~c_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        if attention_mask is not None:
            if attention_mask.ndim == 2:
                scores = scores.masked_fill(~(attention_mask.unsqueeze(1).unsqueeze(2) > 0), float("-inf"))
            elif attention_mask.ndim == 4:
                scores = scores.masked_fill(~(attention_mask > 0), float("-inf"))

        k_val = min(self.top_k, N)
        topk_scores, topk_indices = torch.topk(scores, k=k_val, dim=-1)

        weights = F.softmax(topk_scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        weights = self.dropout(weights)

        # Gather values matching top-k indices: [B, H, N, k_val, head_dim]
        v_expanded = v.unsqueeze(2).expand(-1, -1, N, -1, -1)
        gather_idx = topk_indices.unsqueeze(-1).expand(-1, -1, -1, -1, self.head_dim)
        gathered_v = torch.gather(v_expanded, dim=3, index=gather_idx)

        context = (weights.unsqueeze(-1) * gathered_v).sum(dim=3)
        out = context.transpose(1, 2).contiguous().view(B, N, D)
        out = self.out_proj(out)
        return out, None
