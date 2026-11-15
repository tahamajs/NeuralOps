#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
The Hedgehog: Learnable Linear Attention via Monotonic MLPs.

Implements the Hedgehog feature map parameterization for linear attention,
training lightweight MLPs to learn expressive non-negative feature representations.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.base import BaseAttention
from neuralops.core.registry import ATTENTION_REGISTRY

logger = logging.getLogger(__name__)


@ATTENTION_REGISTRY.register(
    "hedgehog",
    supports_causal=True,
    supports_bidirectional=True,
    supports_gqa=False,
)
@ATTENTION_REGISTRY.register(
    "hedgehog_attention",
    is_alias=True,
    supports_causal=True,
    supports_bidirectional=True,
    supports_gqa=False,
)
class HedgehogAttention(BaseAttention):
    """Hedgehog linear attention with learnable monotonic feature maps.

    Attributes:
        FAMILY: Taxonomy family identifier ('linear').
        supports_causal: Whether causal linear recurrence is supported.
    """

    FAMILY: str = "linear"
    supports_causal: bool = True
    supports_bidirectional: bool = True

    def __init__(
        self,
        config: Any | None = None,
        dim: Optional[int] = None,
        num_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        feature_dim: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize Hedgehog attention.

        Args:
            config: Optional configuration object.
            dim: Total model dimension.
            num_heads: Number of attention heads.
            head_dim: Feature dimension per head.
            feature_dim: Transformed feature dimension for queries and keys.
            **kwargs: Extra arguments forwarded to BaseAttention.
        """
        super().__init__(config=config, **kwargs)

        resolved_dim = (
            dim
            or kwargs.get("hidden_size")
            or kwargs.get("d_model")
            or (getattr(config, "hidden_size", getattr(config, "dim", 128)) if config is not None else 128)
        )
        self.dim = int(resolved_dim)
        self.hidden_size = self.dim

        resolved_heads = (
            num_heads
            or kwargs.get("num_attention_heads")
            or (getattr(config, "num_attention_heads", getattr(config, "num_heads", 4)) if config is not None else 4)
        )
        self.num_heads = int(resolved_heads)
        self.head_dim = head_dim or (self.dim // self.num_heads)
        self.feature_dim = feature_dim or self.head_dim

        self.q_proj = nn.Linear(self.dim, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.dim, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.dim, self.num_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, self.dim, bias=False)

        # Monotonic MLP parameterizations for Q and K
        self.q_mlp = nn.Sequential(
            nn.Linear(self.head_dim, self.feature_dim, bias=True),
            nn.GELU(),
            nn.Linear(self.feature_dim, self.feature_dim, bias=True),
        )
        self.k_mlp = nn.Sequential(
            nn.Linear(self.head_dim, self.feature_dim, bias=True),
            nn.GELU(),
            nn.Linear(self.feature_dim, self.feature_dim, bias=True),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize projection and MLP weights."""
        nn.init.xavier_normal_(self.q_proj.weight)
        nn.init.xavier_normal_(self.k_proj.weight)
        nn.init.xavier_normal_(self.v_proj.weight)
        nn.init.xavier_normal_(self.out_proj.weight)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass computing causal Hedgehog linear attention.

        Args:
            hidden_states: Input tensor of shape (B, N, D).
            attention_mask: Optional attention mask.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Tuple of (output tensor of shape (B, N, D), None).
        """
        B, N, _ = hidden_states.shape
        H = self.num_heads
        d = self.head_dim

        q = self.q_proj(hidden_states).view(B, N, H, d)
        k = self.k_proj(hidden_states).view(B, N, H, d)
        v = self.v_proj(hidden_states).view(B, N, H, d)

        # Apply learnable monotonic feature maps: phi(x) = exp(clamp(MLP(x), max=10.0))
        phi_q = torch.exp(torch.clamp(self.q_mlp(q), max=10.0)).transpose(1, 2)  # (B, H, N, d_feat)
        phi_k = torch.exp(torch.clamp(self.k_mlp(k), max=10.0)).transpose(1, 2)  # (B, H, N, d_feat)
        v_h = v.transpose(1, 2)  # (B, H, N, d)

        # Causal cumulative prefix scan
        kv = torch.einsum("bhnd,bhnv->bhndv", phi_k, v_h)
        s_accum = torch.cumsum(kv, dim=2)
        z_accum = torch.cumsum(phi_k, dim=2)

        num = torch.einsum("bhnd,bhndv->bhnv", phi_q, s_accum)
        den = torch.einsum("bhnd,bhnd->bhn", phi_q, z_accum).unsqueeze(-1).clamp(min=1e-5)

        out = num / den
        out = out.transpose(1, 2).contiguous().view(B, N, H * d)
        out = self.out_proj(out)
        return out, None


__all__ = ["HedgehogAttention"]
