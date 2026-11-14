#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
RWKV Attention Layer (RWKV-v6 Receptance Weighted Key Value)

Implements the linear-complexity recurrent attention mechanism from RWKV-v6,
supporting linear-time autoregressive decoding and efficient parallel training.
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
    "rwkv",
    supports_causal=True,
    supports_bidirectional=False,
    supports_gqa=False,
)
@ATTENTION_REGISTRY.register(
    "rwkv6",
    is_alias=True,
    supports_causal=True,
    supports_bidirectional=False,
    supports_gqa=False,
)
@ATTENTION_REGISTRY.register(
    "rwkv_attention",
    is_alias=True,
    supports_causal=True,
    supports_bidirectional=False,
    supports_gqa=False,
)
class RWKVAttention(BaseAttention):
    """RWKV-v6 Receptance Weighted Key Value recurrent attention layer.

    Attributes:
        supports_causal: Whether causal masking is supported (always True for RWKV).
        FAMILY: Taxonomy family identifier ('ssm').
    """

    FAMILY: str = "ssm"
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
        """Initialize the RWKV attention module.

        Args:
            config: Optional configuration object providing hidden_size and num_heads.
            dim: Model hidden dimension.
            num_heads: Number of attention heads.
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

        # Projections for Receptance, Key, Value, Gate / Decay
        self.r_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.k_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.v_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.w_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.out_proj = nn.Linear(self.dim, self.dim, bias=False)

        # Per-head bonus parameter
        self.u = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize projection weights and bonus vector with safe bounds."""
        nn.init.orthogonal_(self.r_proj.weight, gain=0.5)
        nn.init.orthogonal_(self.k_proj.weight, gain=0.5)
        nn.init.orthogonal_(self.v_proj.weight, gain=0.5)
        nn.init.zeros_(self.w_proj.weight)
        nn.init.orthogonal_(self.out_proj.weight, gain=0.5)
        nn.init.normal_(self.u, mean=0.0, std=0.02)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass for RWKV recurrence.

        Args:
            hidden_states: Input tensor of shape (B, N, D) or (B, H, N, D).
            attention_mask: Optional attention mask (unused in pure causal scan).
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple of (output tensor of same shape as input, None).
        """
        from neuralops.ops.triton.rwkv6_scan import rwkv6_scan_pytorch

        if hidden_states.dim() == 4:
            # Direct 4D tensor input (B, H, N, D)
            b, h, n, d = hidden_states.shape
            r = hidden_states
            k = hidden_states
            v = hidden_states
            w = torch.full_like(r, -0.5)
            u = self.u if self.u.shape == (h, d) else torch.zeros(h, d, device=hidden_states.device)
            out = rwkv6_scan_pytorch(r, k, v, w, u)
            return out, None

        b, n, d = hidden_states.shape
        h = self.num_heads
        head_d = self.head_dim

        r = self.r_proj(hidden_states).view(b, n, h, head_d).transpose(1, 2)
        k = self.k_proj(hidden_states).view(b, n, h, head_d).transpose(1, 2)
        v = self.v_proj(hidden_states).view(b, n, h, head_d).transpose(1, 2)
        # Decay is strictly negative
        w_raw = self.w_proj(hidden_states).view(b, n, h, head_d).transpose(1, 2)
        w = -torch.exp(torch.clamp(w_raw, max=5.0))

        u = self.u
        if u.device != hidden_states.device:
            u = u.to(hidden_states.device)

        out = rwkv6_scan_pytorch(r, k, v, w, u)
        out = out.transpose(1, 2).contiguous().view(b, n, d)
        out = self.out_proj(out)
        return out, None


__all__ = ["RWKVAttention"]
