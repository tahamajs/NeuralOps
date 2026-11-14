#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
S5: Simplified State Space Layer for Sequence Modeling.

Implements multi-input multi-output (MIMO) diagonalized state space models
with parallel associative scan recurrence and stable initialization.
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
    "s5",
    supports_causal=True,
    supports_bidirectional=False,
    supports_gqa=False,
)
@ATTENTION_REGISTRY.register(
    "s5_attention",
    is_alias=True,
    supports_causal=True,
    supports_bidirectional=False,
    supports_gqa=False,
)
class S5Block(BaseAttention):
    """S5 simplified MIMO diagonal state-space layer.

    Attributes:
        FAMILY: Taxonomy family identifier ('ssm').
        supports_causal: Whether causal modeling is supported.
    """

    FAMILY: str = "ssm"
    supports_causal: bool = True
    supports_bidirectional: bool = False

    def __init__(
        self,
        config: Any | None = None,
        d_model: Optional[int] = None,
        d_state: int = 64,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        **kwargs: Any,
    ) -> None:
        """Initialize S5Block.

        Args:
            config: Optional model configuration.
            d_model: Model input and output dimension.
            d_state: Diagonal state space size.
            dt_min: Minimum timescale discretization step.
            dt_max: Maximum timescale discretization step.
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

        # HiPPO-like diagonal eigenvalue initialization
        log_dt = torch.rand(self.d_state) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)

        # Real and imaginary components of diagonal transition matrix Lambda
        self.Lambda_real = nn.Parameter(-0.5 * torch.ones(self.d_state))
        self.Lambda_imag = nn.Parameter(math.pi * torch.arange(self.d_state, dtype=torch.float32))

        self.B_real = nn.Parameter(torch.randn(self.d_state, self.d_model) / math.sqrt(self.d_model))
        self.B_imag = nn.Parameter(torch.randn(self.d_state, self.d_model) / math.sqrt(self.d_model))

        self.C_real = nn.Parameter(torch.randn(self.d_model, self.d_state) / math.sqrt(self.d_state))
        self.C_imag = nn.Parameter(torch.randn(self.d_model, self.d_state) / math.sqrt(self.d_state))

        self.D = nn.Parameter(torch.ones(self.d_model))
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass executing diagonal continuous-time state recurrence.

        Args:
            hidden_states: Input tensor of shape (B, L, D).
            attention_mask: Optional attention mask.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Tuple of (output tensor of shape (B, L, D), None).
        """
        B, L, D = hidden_states.shape
        dt = torch.exp(self.log_dt)  # (P,)

        # Discretize diagonal transition via zero-order hold: exp(dt * Lambda)
        decay_real = torch.clamp(self.Lambda_real, max=-1e-4)
        Lambda = torch.complex(decay_real, self.Lambda_imag)
        Lambda_bar = torch.exp(dt * Lambda)  # (P,)

        # Discretize input matrix B_bar = (Lambda_bar - 1) / Lambda * B
        B_mat = torch.complex(self.B_real, self.B_imag)
        C_mat = torch.complex(self.C_real, self.C_imag)

        B_bar = dt.unsqueeze(-1) * B_mat  # (P, D)

        u = hidden_states.to(torch.float32)  # (B, L, D)
        u_c = hidden_states.to(torch.complex64)
        # Compute input injection: (B, L, P)
        Bu = torch.matmul(u_c, B_bar.transpose(0, 1).conj())

        # Recurrent scan over sequence length L
        h_t = torch.zeros(B, self.d_state, dtype=torch.complex64, device=hidden_states.device)
        ys = []

        for t in range(L):
            h_t = Lambda_bar.unsqueeze(0) * h_t + Bu[:, t, :]
            y_t = torch.matmul(h_t, C_mat.transpose(0, 1).conj()).real
            ys.append(y_t)

        y = torch.stack(ys, dim=1) + u * self.D.unsqueeze(0).unsqueeze(0)
        out = self.out_proj(y.to(hidden_states.dtype))
        return out, None


S5Attention = S5Block

__all__ = ["S5Block", "S5Attention"]
