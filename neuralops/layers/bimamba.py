#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
BiMamba: Bidirectional Mamba Sequence Modeling Layer.

Executes dual selective state space models across forward and time-reversed sequences,
enabling non-causal representation learning for vision and encoder backbones.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
import torch.nn as nn

from neuralops.core.base import BaseAttention
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.layers.ssm.mamba_block import MambaBlock

logger = logging.getLogger(__name__)


@ATTENTION_REGISTRY.register(
    "bimamba",
    supports_causal=False,
    supports_bidirectional=True,
    supports_gqa=False,
)
@ATTENTION_REGISTRY.register(
    "bimamba_block",
    is_alias=True,
    supports_causal=False,
    supports_bidirectional=True,
    supports_gqa=False,
)
class BiMambaBlock(BaseAttention):
    """Bidirectional Mamba sequence processing block.

    Attributes:
        FAMILY: Taxonomy family identifier ('ssm').
        supports_bidirectional: Whether bidirectional modeling is supported.
    """

    FAMILY: str = "ssm"
    supports_causal: bool = False
    supports_bidirectional: bool = True

    def __init__(
        self,
        config: Any | None = None,
        d_model: Optional[int] = None,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        **kwargs: Any,
    ) -> None:
        """Initialize BiMambaBlock.

        Args:
            config: Optional configuration.
            d_model: Model hidden dimension.
            d_state: SSM latent state dimension.
            d_conv: 1D convolution kernel size.
            expand: Hidden expansion factor.
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

        self.fwd_mamba = MambaBlock(
            config=config,
            d_model=self.d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            bidirectional=False,
            **kwargs,
        )
        self.bwd_mamba = MambaBlock(
            config=config,
            d_model=self.d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            bidirectional=False,
            **kwargs,
        )
        self.out_proj = nn.Linear(2 * self.d_model, self.d_model, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Bidirectional forward pass.

        Args:
            hidden_states: Input tensor of shape (B, L, D).
            attention_mask: Optional attention mask.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Tuple of (output tensor of shape (B, L, D), None).
        """
        out_fwd, _ = self.fwd_mamba(hidden_states, attention_mask=attention_mask, **kwargs)

        # Reverse sequence for backward pass
        rev_states = torch.flip(hidden_states, dims=[1])
        rev_mask = torch.flip(attention_mask, dims=[1]) if attention_mask is not None else None
        out_bwd_rev, _ = self.bwd_mamba(rev_states, attention_mask=rev_mask, **kwargs)
        out_bwd = torch.flip(out_bwd_rev, dims=[1])

        combined = torch.cat([out_fwd, out_bwd], dim=-1)
        out = self.out_proj(combined)
        return out, None


BiMambaAttention = BiMambaBlock

__all__ = ["BiMambaBlock", "BiMambaAttention"]
