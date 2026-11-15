#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Fast Linear Attention layer with recurrent state update."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.base import BaseAttention
from neuralops.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register(
    "fast_attention",
    supports_causal=True,
    supports_bidirectional=True,
    supports_gqa=True,
)
class FastAttention(BaseAttention):
    """Fast linear attention using feature map phi(x) = elu(x) + 1."""

    supports_causal: bool = True
    supports_bidirectional: bool = True
    supports_gqa: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,
        dim: int | None = None,
        num_heads: int = 4,
        is_causal: bool = True,
        dropout: float = 0.0,
        **kwargs: Any,
    ) -> None:
        """Initialize FastAttention.

        Args:
            config: Optional config.
            dim: Embedding dimension.
            num_heads: Attention heads.
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
        self.is_causal = bool(kwargs.get("is_causal", is_causal))

        self.q_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.k_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.v_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.out_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.dropout = nn.Dropout(float(dropout))

    @staticmethod
    def _feature_map(x: torch.Tensor) -> torch.Tensor:
        """Non-negative kernel mapping phi(x) = elu(x) + 1."""
        return F.elu(x) + 1.0

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        """Forward pass for FastAttention.

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

        q_prime = self._feature_map(q)
        k_prime = self._feature_map(k)

        if attention_mask is not None:
            if attention_mask.ndim == 2:
                k_prime = k_prime * (attention_mask.unsqueeze(1).unsqueeze(-1) > 0)
            elif attention_mask.ndim == 4:
                k_prime = k_prime * (attention_mask.mean(dim=2, keepdim=True).transpose(-2, -1) > 0)

        if self.is_causal:
            kv = torch.einsum("bhnd,bhnm->bhndm", k_prime, v)
            kv_cum = torch.cumsum(kv, dim=2)
            k_cum = torch.cumsum(k_prime, dim=2)
            context = torch.einsum("bhnd,bhndm->bhnm", q_prime, kv_cum)
            norm = (q_prime * k_cum).sum(dim=-1, keepdim=True).clamp(min=1e-6)
            context = context / norm
        else:
            kv = torch.einsum("bhnd,bhnm->bhdm", k_prime, v)
            k_sum = k_prime.sum(dim=2, keepdim=True)
            context = torch.einsum("bhnd,bhdm->bhnm", q_prime, kv)
            norm = (q_prime * k_sum).sum(dim=-1, keepdim=True).clamp(min=1e-6)
            context = context / norm

        context = torch.nan_to_num(context, nan=0.0)
        context = self.dropout(context)
        out = context.transpose(1, 2).contiguous().view(B, N, D)
        out = self.out_proj(out)
        return out, None
