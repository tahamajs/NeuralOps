#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Based Attention Layer (2nd-Order Taylor Expansion Linear Attention).

Implements the Based linear attention architecture balancing recall and throughput
via 2nd-order polynomial kernel expansion.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional

import torch
import torch.nn as nn

from neuralops.core.base import BaseAttention
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.ops.triton.based_scan import based_taylor_scan, based_taylor_scan_pytorch

logger = logging.getLogger(__name__)


@ATTENTION_REGISTRY.register(
    "based",
    supports_causal=True,
    supports_bidirectional=False,
    supports_gqa=False,
)
@ATTENTION_REGISTRY.register(
    "based_attention",
    is_alias=True,
    supports_causal=True,
    supports_bidirectional=False,
    supports_gqa=False,
)
class BasedAttention(BaseAttention):
    """Based 2nd-order Taylor expansion linear attention layer.

    Attributes:
        FAMILY: Taxonomy family identifier ('linear').
        supports_causal: Whether causal masking is supported (always True).
    """

    FAMILY: str = "linear"
    supports_causal: bool = True
    supports_bidirectional: bool = False

    def __init__(
        self,
        config: Any | None = None,
        dim: Optional[int] = None,
        num_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        feature_dim: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize Based Attention.

        Args:
            config: Optional model configuration.
            dim: Model hidden dimension.
            num_heads: Number of attention heads.
            head_dim: Per-head dimension.
            feature_dim: Optional Taylor feature projection dimension.
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
        """Forward pass for Based linear attention.

        Args:
            hidden_states: Input tensor of shape (B, N, D) or (B, H, N, D).
            attention_mask: Optional attention mask.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Tuple of (output tensor of same shape as input, None).
        """
        if hidden_states.dim() == 4:
            b, h, n, d = hidden_states.shape
            out = based_taylor_scan(hidden_states, hidden_states, hidden_states, scale=self.scale)
            return out, None

        b, n, _ = hidden_states.shape
        h = self.num_heads
        d = self.head_dim

        q = self.q_proj(hidden_states).view(b, n, h, d).transpose(1, 2)
        k = self.k_proj(hidden_states).view(b, n, h, d).transpose(1, 2)
        v = self.v_proj(hidden_states).view(b, n, h, d).transpose(1, 2)

        out = based_taylor_scan(q, k, v, scale=self.scale)
        out = out.transpose(1, 2).contiguous().view(b, n, h * d)
        out = self.out_proj(out)
        return out, None


__all__ = ["BasedAttention"]
