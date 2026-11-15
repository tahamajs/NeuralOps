#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Sparsemax Attention layer using Euclidean projection onto the probability simplex."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from neuralops.core.base import BaseAttention
from neuralops.core.registry import ATTENTION_REGISTRY


def sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Compute sparsemax activation function.

    Computes Euclidean projection onto the simplex, yielding exact zero probabilities for small logits.
    """
    sorted_logits, _ = torch.sort(logits, descending=True, dim=dim)
    k_range = torch.arange(1, sorted_logits.size(dim) + 1, device=logits.device, dtype=logits.dtype)
    shape = [1] * logits.ndim
    shape[dim] = sorted_logits.size(dim)
    k_range = k_range.view(shape)

    cumsum = torch.cumsum(sorted_logits, dim=dim)
    support = 1.0 + k_range * sorted_logits > cumsum
    k_z = support.sum(dim=dim, keepdim=True).clamp(min=1)
    tau_z = (cumsum.gather(dim=dim, index=k_z.long() - 1) - 1.0) / k_z
    return torch.clamp(logits - tau_z, min=0.0)


@ATTENTION_REGISTRY.register(
    "sparsemax",
    supports_causal=True,
    supports_bidirectional=True,
    supports_gqa=True,
)
@ATTENTION_REGISTRY.register(
    "sparsemax_attention",
    is_alias=True,
    supports_causal=True,
    supports_bidirectional=True,
    supports_gqa=True,
)
class SparsemaxAttention(BaseAttention):
    """Attention layer using sparsemax instead of softmax for exact sparse attention weights."""

    supports_causal: bool = True
    supports_bidirectional: bool = True
    supports_gqa: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,
        dim: int | None = None,
        num_heads: int = 4,
        is_causal: bool = False,
        dropout: float = 0.0,
        **kwargs: Any,
    ) -> None:
        """Initialize SparsemaxAttention.

        Args:
            config: Optional config.
            dim: Embedding dimension.
            num_heads: Head count.
            is_causal: Autoregressive flag.
            dropout: Dropout probability.
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
        """Forward pass for SparsemaxAttention.

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

        weights = sparsemax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        weights = self.dropout(weights)

        context = torch.matmul(weights, v)
        out = context.transpose(1, 2).contiguous().view(B, N, D)
        out = self.out_proj(out)
        return out, None
