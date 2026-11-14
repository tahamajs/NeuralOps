#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Linear Recurrent Unit (LRU).

Implements DeepMind's Linear Recurrent Unit (LRU) architecture featuring complex-diagonal
parameterized recurrent state transitions with provable stability bounds.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional

import torch
import torch.nn as nn

from neuralops.core.base import BaseAttention
from neuralops.core.registry import ATTENTION_REGISTRY

logger = logging.getLogger(__name__)


@ATTENTION_REGISTRY.register(
    "lru",
    supports_causal=True,
    supports_bidirectional=False,
    supports_gqa=False,
)
@ATTENTION_REGISTRY.register(
    "lru_attention",
    is_alias=True,
    supports_causal=True,
    supports_bidirectional=False,
    supports_gqa=False,
)
class LRUAttention(BaseAttention):
    """Linear Recurrent Unit layer with stable complex eigenvalue parametrization.

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
        d_state: Optional[int] = None,
        r_min: float = 0.9,
        r_max: float = 0.999,
        max_phase: float = 2.0 * math.pi,
        **kwargs: Any,
    ) -> None:
        """Initialize LRU layer.

        Args:
            config: Optional model configuration.
            d_model: Input and output feature dimension.
            d_state: State vector dimension. Defaults to d_model.
            r_min: Minimum magnitude of eigenvalue radius.
            r_max: Maximum magnitude of eigenvalue radius.
            max_phase: Maximum phase angle for complex rotation.
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
        self.d_state = int(d_state or self.d_model)

        # Parameterize ring radius: r = exp(-exp(nu_log)) in (r_min, r_max)
        u_rand = torch.rand(self.d_state)
        nu = -torch.log(r_min + (r_max - r_min) * u_rand)
        self.nu_log = nn.Parameter(torch.log(nu))

        # Parameterize phase angle
        self.theta_log = nn.Parameter(torch.log(torch.rand(self.d_state) * max_phase))

        self.B_real = nn.Parameter(torch.randn(self.d_state, self.d_model) / math.sqrt(2 * self.d_model))
        self.B_imag = nn.Parameter(torch.randn(self.d_state, self.d_model) / math.sqrt(2 * self.d_model))
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
        """Forward pass through LRU recurrent state updates.

        Args:
            hidden_states: Input tensor of shape (B, L, D).
            attention_mask: Optional attention mask.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Tuple of (output tensor of shape (B, L, D), None).
        """
        B, L, D = hidden_states.shape

        # Construct complex eigenvalues Lambda = exp(-nu + i * theta)
        nu = torch.exp(self.nu_log)
        theta = torch.exp(self.theta_log)
        Lambda = torch.exp(torch.complex(-nu, theta))

        # Normalization factor gamma = sqrt(1 - |Lambda|^2)
        gamma = torch.sqrt(torch.clamp(1.0 - torch.abs(Lambda) ** 2, min=1e-8))

        B_mat = torch.complex(self.B_real, self.B_imag) * gamma.unsqueeze(-1)
        C_mat = torch.complex(self.C_real, self.C_imag)

        u = hidden_states.to(torch.float32)
        u_c = hidden_states.to(torch.complex64)
        Bu = torch.matmul(u_c, B_mat.transpose(0, 1).conj())

        h_t = torch.zeros(B, self.d_state, dtype=torch.complex64, device=hidden_states.device)
        ys = []

        for t in range(L):
            h_t = Lambda.unsqueeze(0) * h_t + Bu[:, t, :]
            y_t = torch.matmul(h_t, C_mat.transpose(0, 1).conj()).real
            ys.append(y_t)

        y = torch.stack(ys, dim=1) + u * self.D.unsqueeze(0).unsqueeze(0)
        out = self.out_proj(y.to(hidden_states.dtype))
        return out, None


LinearRecurrentUnit = LRUAttention

__all__ = ["LRUAttention", "LinearRecurrentUnit"]
