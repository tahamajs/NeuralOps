#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
RetNet Attention Layer (Multi-Scale Retention)

Implements Multi-Scale Retention (RetNet), achieving the 'impossible triangle':
training parallelism, linear-complexity inference, and state-of-the-art performance.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from neuralops.core.base import BaseAttention
from neuralops.core.registry import ATTENTION_REGISTRY

logger = logging.getLogger(__name__)


@ATTENTION_REGISTRY.register(
    "retnet",
    supports_causal=True,
    supports_bidirectional=False,
    supports_gqa=False,
)
@ATTENTION_REGISTRY.register(
    "retnet_attention",
    is_alias=True,
    supports_causal=True,
    supports_bidirectional=False,
    supports_gqa=False,
)
class RetNetAttention(BaseAttention):
    """RetNet Multi-Scale Retention attention layer.

    Attributes:
        supports_causal: Whether causal masking is supported (always True for RetNet).
        FAMILY: Taxonomy family identifier ('linear').
    """

    FAMILY: str = "linear"
    supports_causal: bool = True
    supports_bidirectional: bool = False

    def __init__(
        self,
        config: Any | None = None,
        dim: int | None = None,
        num_heads: int | None = None,
        head_dim: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the RetNet attention module.

        Args:
            config: Optional configuration object providing hidden_size and num_heads.
            dim: Model hidden dimension.
            num_heads: Number of retention heads.
            head_dim: Per-head dimension.
            **kwargs: Additional parameters forwarded to BaseAttention.
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

        self.q_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.k_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.v_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.out_proj = nn.Linear(self.dim, self.dim, bias=False)

        # Multi-scale exponential decay factor per head: gamma = 1 - 2^(-5 - h)
        gammas = 1.0 - 2.0 ** (-5.0 - torch.arange(self.num_heads, dtype=torch.float32))
        self.register_buffer("gammas", gammas)
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize projection weights with normalized Xavier normal."""
        nn.init.xavier_normal_(self.q_proj.weight)
        nn.init.xavier_normal_(self.k_proj.weight)
        nn.init.xavier_normal_(self.v_proj.weight)
        nn.init.xavier_normal_(self.out_proj.weight)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass for multi-scale retention.

        Args:
            hidden_states: Input tensor of shape (B, N, D) or (B, H, N, D).
            attention_mask: Optional attention mask.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple of (output tensor of same shape as input, None).
        """
        from neuralops.ops.triton.retnet_retention import triton_multi_scale_retention_pytorch_ref

        if hidden_states.dim() == 4:
            # Direct 4D tensor input (B, H, N, D)
            b, h, n, d = hidden_states.shape
            gammas = self.gammas.to(hidden_states.device)
            if gammas.shape[0] != h:
                gammas = 1.0 - 2.0 ** (-5.0 - torch.arange(h, dtype=torch.float32, device=hidden_states.device))
            out = triton_multi_scale_retention_pytorch_ref(hidden_states, hidden_states, hidden_states, gammas)
            return out, None

        b, n, d = hidden_states.shape
        h = self.num_heads
        head_d = self.head_dim

        q = self.q_proj(hidden_states).view(b, n, h, head_d).transpose(1, 2)
        k = self.k_proj(hidden_states).view(b, n, h, head_d).transpose(1, 2)
        v = self.v_proj(hidden_states).view(b, n, h, head_d).transpose(1, 2)

        gammas = self.gammas.to(hidden_states.device)
        out = triton_multi_scale_retention_pytorch_ref(q, k, v, gammas)
        out = out.transpose(1, 2).contiguous().view(b, n, d)
        out = self.out_proj(out)
        return out, None


__all__ = ["RetNetAttention"]
