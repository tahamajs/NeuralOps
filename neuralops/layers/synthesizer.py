#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Synthesizer attention layer with synthetic attention weights (Tay et al., 2021)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.base import BaseAttention
from neuralops.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register(
    "synthesizer",
    supports_causal=True,
    supports_bidirectional=True,
    supports_gqa=True,
)
@ATTENTION_REGISTRY.register(
    "synthesizer_attention",
    is_alias=True,
    supports_causal=True,
    supports_bidirectional=True,
    supports_gqa=True,
)
class SynthesizerAttention(BaseAttention):
    """Dense Synthesizer Attention predicting attention weights directly without pairwise dot products."""

    supports_causal: bool = True
    supports_bidirectional: bool = True
    supports_gqa: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,
        max_seq_len: int = 512,
        dim: int | None = None,
        num_heads: int = 4,
        is_causal: bool = False,
        dropout: float = 0.0,
        **kwargs: Any,
    ) -> None:
        """Initialize SynthesizerAttention.

        Args:
            config: Optional config.
            max_seq_len: Max sequence length for synthetic matrix.
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

        self.max_seq_len = max(1, int(kwargs.get("max_seq_len", max_seq_len)))
        self.is_causal = bool(kwargs.get("is_causal", is_causal))

        self.w1 = nn.Linear(self.dim, self.max_seq_len)
        self.v_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.out_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.dropout = nn.Dropout(float(dropout))

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        """Forward pass for SynthesizerAttention.

        Args:
            x: Input tensor [B, N, D].
            attention_mask: Optional padding mask.
            **kwargs: Extra parameters.

        Returns:
            Tuple of (output, None).
        """
        B, N, D = x.shape
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Predict synthetic energy logits directly from query token representations
        logits = self.w1(x)[:, :, :N]  # [B, N, N]
        scores = logits.unsqueeze(1).expand(-1, self.num_heads, -1, -1)  # [B, H, N, N]

        if self.is_causal:
            idx = torch.arange(N, device=x.device)
            c_mask = idx.unsqueeze(1) >= idx.unsqueeze(0)
            scores = scores.masked_fill(~c_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        if attention_mask is not None:
            if attention_mask.ndim == 2:
                scores = scores.masked_fill(~(attention_mask.unsqueeze(1).unsqueeze(2) > 0), float("-inf"))
            elif attention_mask.ndim == 4:
                scores = scores.masked_fill(~(attention_mask > 0), float("-inf"))

        weights = F.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        weights = self.dropout(weights)

        context = torch.matmul(weights, v)
        out = context.transpose(1, 2).contiguous().view(B, N, D)
        out = self.out_proj(out)
        return out, None
