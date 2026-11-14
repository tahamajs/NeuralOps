#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Time-to-Recurrence Attention (T2R).

Transforms position intervals into time-decayed recurrent state spaces,
enabling continuous-time positional awareness in linear attention mechanisms.
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
    "t2r",
    supports_causal=True,
    supports_bidirectional=False,
    supports_gqa=False,
)
@ATTENTION_REGISTRY.register(
    "t2r_attention",
    is_alias=True,
    supports_causal=True,
    supports_bidirectional=False,
    supports_gqa=False,
)
class T2RAttention(BaseAttention):
    """Time-to-Recurrence linear attention layer.

    Attributes:
        FAMILY: Taxonomy family identifier ('linear').
        supports_causal: Whether causal linear recurrence is supported.
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
        **kwargs: Any,
    ) -> None:
        """Initialize T2RAttention module.

        Args:
            config: Optional configuration object.
            dim: Total model hidden dimension.
            num_heads: Number of attention heads.
            head_dim: Feature dimension per head.
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

        self.q_proj = nn.Linear(self.dim, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.dim, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.dim, self.num_heads * self.head_dim, bias=False)
        self.decay_proj = nn.Linear(self.dim, self.num_heads, bias=True)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, self.dim, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with standard normal distribution."""
        nn.init.xavier_normal_(self.q_proj.weight)
        nn.init.xavier_normal_(self.k_proj.weight)
        nn.init.xavier_normal_(self.v_proj.weight)
        nn.init.xavier_normal_(self.out_proj.weight)
        nn.init.constant_(self.decay_proj.bias, 2.0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through T2R attention.

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

        q = F.elu(self.q_proj(hidden_states).view(B, N, H, d)) + 1.0
        k = F.elu(self.k_proj(hidden_states).view(B, N, H, d)) + 1.0
        v = self.v_proj(hidden_states).view(B, N, H, d)

        # Compute per-head time decay factors: gamma in (0, 1)
        decay = torch.sigmoid(self.decay_proj(hidden_states)).view(B, N, H, 1)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        decay = decay.transpose(1, 2)

        # Recurrent state formulation with data-dependent decay
        s_t = torch.zeros(B, H, d, d, device=hidden_states.device, dtype=hidden_states.dtype)
        z_t = torch.zeros(B, H, d, 1, device=hidden_states.device, dtype=hidden_states.dtype)
        ys = []

        for t in range(N):
            gamma_t = decay[:, :, t, :].unsqueeze(-1)
            q_t = q[:, :, t, :].unsqueeze(-2)  # (B, H, 1, d)
            k_t = k[:, :, t, :].unsqueeze(-1)  # (B, H, d, 1)
            v_t = v[:, :, t, :].unsqueeze(-2)  # (B, H, 1, d)

            s_t = gamma_t * s_t + torch.matmul(k_t, v_t)
            z_t = gamma_t * z_t + k_t

            num = torch.matmul(q_t, s_t).squeeze(-2)
            den = torch.matmul(q_t, z_t).squeeze(-2).clamp(min=1e-5)
            y_t = num / den
            ys.append(y_t)

        out = torch.stack(ys, dim=2).transpose(1, 2).contiguous().view(B, N, H * d)
        out = self.out_proj(out)
        return out, None


__all__ = ["T2RAttention"]
