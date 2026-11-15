#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""xFormers memory-efficient block sparse attention layer."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.base import BaseAttention
from neuralops.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register(
    "xformers_attention",
    supports_causal=True,
    supports_bidirectional=True,
    supports_gqa=True,
)
@ATTENTION_REGISTRY.register(
    "xformers",
    is_alias=True,
    supports_causal=True,
    supports_bidirectional=True,
    supports_gqa=True,
)
class XFormersAttention(BaseAttention):
    """Memory-efficient sparse attention inspired by xFormers."""

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
        """Initialize XFormersAttention.

        Args:
            config: Optional config.
            dim: Embedding dimension.
            num_heads: Number of attention heads.
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
        self.dropout_p = float(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        """Forward pass for XFormersAttention.

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

        # Utilize native SDPA when available with memory-efficient backend dispatch
        if attention_mask is None:
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=self.is_causal,
            )
        else:
            mask = attention_mask
            if mask.ndim == 2:
                mask = mask.unsqueeze(1).unsqueeze(2) > 0
            elif mask.ndim == 4:
                mask = mask > 0

            if self.is_causal:
                idx = torch.arange(N, device=x.device)
                c_mask = (idx.unsqueeze(1) >= idx.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
                mask = mask & c_mask

            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=False,
            )

        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.out_proj(out)
        return out, None
