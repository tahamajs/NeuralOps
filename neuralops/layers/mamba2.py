#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Mamba-2 Layer (State Space Duality / SSD Architecture).

Implements the Mamba-2 block combining 1D depthwise convolution with
multi-head State Space Duality (SSD) structured semi-separable matrix transformations.
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
from neuralops.ops.cuda.mamba2_block.mamba2_block_autograd import (
    mamba2_block,
    mamba2_block_ref,
)

logger = logging.getLogger(__name__)


@ATTENTION_REGISTRY.register(
    "mamba2",
    supports_causal=True,
    supports_bidirectional=False,
    supports_gqa=False,
)
@ATTENTION_REGISTRY.register(
    "mamba2_block",
    is_alias=True,
    supports_causal=True,
    supports_bidirectional=False,
    supports_gqa=False,
)
class Mamba2Block(BaseAttention):
    """Mamba-2 State Space Duality (SSD) layer.

    Attributes:
        FAMILY: Taxonomy family identifier ('ssm').
        supports_causal: Whether causal sequence modeling is supported.
    """

    FAMILY: str = "ssm"
    supports_causal: bool = True
    supports_bidirectional: bool = False

    def __init__(
        self,
        config: Any | None = None,
        d_model: Optional[int] = None,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        **kwargs: Any,
    ) -> None:
        """Initialize Mamba2Block.

        Args:
            config: Optional configuration object.
            d_model: Input/output feature dimension.
            d_state: SSM latent state dimension N.
            d_conv: 1D convolution kernel width.
            expand: Inner expansion factor.
            headdim: Per-head dimension.
            **kwargs: Extra arguments forwarded to BaseAttention.
        """
        super().__init__(config=config, **kwargs)

        resolved_dim = (
            d_model
            or kwargs.get("hidden_size")
            or kwargs.get("dim")
            or (getattr(config, "hidden_size", getattr(config, "d_model", 128)) if config is not None else 128)
        )
        self.d_model = int(resolved_dim)
        self.hidden_size = self.d_model
        self.d_state = int(d_state)
        self.d_conv = int(d_conv)
        self.expand = int(expand)
        self.d_inner = self.expand * self.d_model
        self.headdim = int(headdim)

        self.in_proj = nn.Linear(self.d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=self.d_conv,
            groups=self.d_inner,
            padding=self.d_conv - 1,
            bias=True,
        )

        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        self.b_proj = nn.Linear(self.d_inner, self.d_state, bias=False)
        self.c_proj = nn.Linear(self.d_inner, self.d_state, bias=False)

        # Log of diagonal decay parameter A
        self.A_log = nn.Parameter(torch.log(torch.arange(1, self.d_inner + 1, dtype=torch.float32)))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize parameters with standard deviations."""
        nn.init.xavier_normal_(self.in_proj.weight)
        nn.init.xavier_normal_(self.out_proj.weight)
        nn.init.zeros_(self.dt_proj.bias)
        nn.init.normal_(self.b_proj.weight, std=0.02)
        nn.init.normal_(self.c_proj.weight, std=0.02)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through Mamba-2 block.

        Args:
            hidden_states: Input tensor of shape (B, L, D).
            attention_mask: Optional attention mask.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Tuple of (output tensor of shape (B, L, D), None).
        """
        B, L, _ = hidden_states.shape
        xz = self.in_proj(hidden_states)
        x, z = xz.chunk(2, dim=-1)

        # Depthwise 1D conv causal padding
        x_conv = self.conv1d(x.transpose(1, 2))[..., :L]
        x_act = F.silu(x_conv).transpose(1, 2)

        dt = F.softplus(self.dt_proj(x_act)).transpose(1, 2)
        b_mat = self.b_proj(x_act).transpose(1, 2)
        c_mat = self.c_proj(x_act).transpose(1, 2)
        u = x_act.transpose(1, 2)
        A = -torch.exp(self.A_log)

        if hidden_states.is_cuda and not torch.is_grad_enabled():
            try:
                y = mamba2_block(u, dt, A, b_mat, c_mat, self.D)
            except Exception:
                y, _ = mamba2_block_ref(u, dt, A, b_mat, c_mat, self.D)
        else:
            y, _ = mamba2_block_ref(u, dt, A, b_mat, c_mat, self.D)

        y = y.transpose(1, 2)
        y_gated = y * F.silu(z)
        out = self.out_proj(y_gated)
        return out, None


Mamba2Attention = Mamba2Block

__all__ = ["Mamba2Block", "Mamba2Attention"]
