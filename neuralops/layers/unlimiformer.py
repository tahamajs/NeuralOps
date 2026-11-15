#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unlimiformer Attention Layer.

Implements the Unlimiformer retrieval-augmented long-range attention architecture,
retrieving top-k keys and values for unbounded sequence processing.
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
    "unlimiformer",
    supports_causal=True,
    supports_bidirectional=True,
    supports_gqa=False,
)
@ATTENTION_REGISTRY.register(
    "unlimiformer_attention",
    is_alias=True,
    supports_causal=True,
    supports_bidirectional=True,
    supports_gqa=False,
)
class UnlimiformerAttention(BaseAttention):
    """Unlimiformer k-NN retrieval-augmented attention layer.

    Attributes:
        FAMILY: Taxonomy family identifier ('linear').
        supports_causal: Whether causal masking is supported.
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
        top_k: int = 64,
        **kwargs: Any,
    ) -> None:
        """Initialize UnlimiformerAttention module.

        Args:
            config: Optional configuration object.
            dim: Model hidden dimension.
            num_heads: Number of attention heads.
            head_dim: Feature dimension per head.
            top_k: Number of nearest neighbors to retrieve per query.
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
        self.top_k = int(top_k)
        self.scale = 1.0 / math.sqrt(max(self.head_dim, 1))

        self.q_proj = nn.Linear(self.dim, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.dim, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.dim, self.num_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, self.dim, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize projection weights."""
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
        """Forward pass using top-k retrieved keys/values.

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

        q = self.q_proj(hidden_states).view(B, N, H, d).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, N, H, d).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, N, H, d).transpose(1, 2)

        # Compute full or top-k attention scores
        k_val = min(self.top_k, N)

        # (B, H, N, N)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                mask = attention_mask.unsqueeze(1).unsqueeze(2)
                scores = scores.masked_fill(mask == 0, -1e4)
            elif attention_mask.dim() == 4:
                scores = scores.masked_fill(attention_mask == 0, -1e4)

        # Causal mask if decoder/causal
        is_causal = getattr(self.config, "is_decoder", False) or getattr(self.config, "is_causal", False)
        if is_causal:
            causal_mask = torch.tril(torch.ones(N, N, device=hidden_states.device, dtype=torch.bool))
            scores = scores.masked_fill(~causal_mask, -1e4)

        if k_val < N:
            # Select top-k keys per query token
            topk_scores, topk_indices = torch.topk(scores, k=k_val, dim=-1)
            probs = F.softmax(topk_scores, dim=-1)

            # Gather values corresponding to top-k indices: (B, H, N, k_val, d)
            v_expanded = v.unsqueeze(2).expand(-1, -1, N, -1, -1)
            gather_idx = topk_indices.unsqueeze(-1).expand(-1, -1, -1, -1, d)
            v_topk = torch.gather(v_expanded, dim=3, index=gather_idx)

            out = torch.matmul(probs.unsqueeze(-2), v_topk).squeeze(-2)
        else:
            probs = F.softmax(scores, dim=-1)
            out = torch.matmul(probs, v)

        out = out.transpose(1, 2).contiguous().view(B, N, H * d)
        out = self.out_proj(out)
        return out, None


__all__ = ["UnlimiformerAttention"]
