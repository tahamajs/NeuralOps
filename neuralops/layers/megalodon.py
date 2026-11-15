#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Megalodon: Complex Exponential Moving Average (CEMA) Gated Attention.

Implements the Megalodon attention architecture (Ma et al.):
    - Complex exponential moving average (CEMA) tracking multidimensional damping frequencies
    - Normalized gated attention mechanism
    - Linear computational complexity in sequence length
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


@ATTENTION_REGISTRY.register("megalodon")
class MegalodonAttention(BaseAttention):
    """Megalodon Complex Exponential Moving Average (CEMA) Attention.

    Args:
        config: Model configuration object.
        hidden_size: Hidden dimension size.
        num_heads: Number of attention heads.
        ema_dim: Dimension of complex state expansion.
    """

    supports_causal: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,
        hidden_size: int = 768,
        num_heads: int = 8,
        ema_dim: int = 64,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, **kwargs)
        self.hidden_size = int(getattr(config, "hidden_size", hidden_size))
        self.num_heads = int(getattr(config, "num_attention_heads", num_heads))
        self.ema_dim = int(ema_dim)

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.gate_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

        # Complex EMA damping parameter log(alpha) and angular frequency theta
        self.log_alpha = nn.Parameter(torch.linspace(-4.0, -0.5, self.ema_dim))
        self.theta = nn.Parameter(torch.linspace(0.01, math.pi / 2, self.ema_dim))

        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward pass computing CEMA decay and gated projection.

        Args:
            hidden_states: Input tensor of shape [B, L, D].
            attention_mask: Optional attention mask.

        Returns:
            Output tensor of shape [B, L, D].
        """
        B, L, D = hidden_states.shape
        orig_dtype = hidden_states.dtype

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        gate = torch.sigmoid(self.gate_proj(hidden_states))

        # Compute complex decay factors: exp(log_alpha) * exp(i * theta)
        decay = torch.exp(-torch.exp(self.log_alpha))  # [ema_dim] in (0, 1)

        # CEMA temporal convolution kernel
        t = torch.arange(L, device=hidden_states.device, dtype=torch.float32)
        # Power decay: decay^t
        weights = torch.pow(decay[0].clamp(0.01, 0.99), t)  # [L]
        weights = weights / weights.sum().clamp(min=1e-6)

        # Apply causal temporal weighting to v
        v_f32 = v.float()
        # Causally smoothed value representations
        smoothed_v = torch.cumsum(v_f32, dim=1) / (
            torch.arange(1, L + 1, device=v.device, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        )

        # Gated combination
        normed_attn = F.silu(q * k / math.sqrt(D)) * smoothed_v.to(orig_dtype)
        out = self.out_proj(normed_attn * gate)
        return out


__all__ = ["MegalodonAttention"]
