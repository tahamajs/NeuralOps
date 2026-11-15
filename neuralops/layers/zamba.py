#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Zamba: Hybrid Mamba-Transformer Architecture Layer.

Implements Zyphra's Zamba architecture combining Mamba state space recurrence
with a global shared Transformer attention block.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
import torch.nn as nn

from neuralops.core.base import BaseAttention
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.layers.dense.full_attention import FullAttention
from neuralops.layers.ssm.mamba_block import MambaBlock

logger = logging.getLogger(__name__)


@ATTENTION_REGISTRY.register("zamba")
class ZambaBlock(BaseAttention):
    """Zamba hybrid layer chaining Mamba SSM and Multi-Head Attention.

    Args:
        config: Model configuration object.
        hidden_size: Hidden dimension size.
        num_heads: Number of attention heads.
        mamba_d_state: State expansion rank for Mamba SSM.
    """

    supports_causal: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,
        hidden_size: int = 768,
        num_heads: int = 12,
        mamba_d_state: int = 16,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, **kwargs)
        self.hidden_size = int(getattr(config, "hidden_size", hidden_size))
        self.num_heads = int(getattr(config, "num_attention_heads", num_heads))

        self.mamba_norm = nn.LayerNorm(self.hidden_size)
        self.mamba = MambaBlock(config=config, hidden_size=self.hidden_size, d_state=mamba_d_state, **kwargs)

        self.attn_norm = nn.LayerNorm(self.hidden_size)
        self.attn = FullAttention(config=config, hidden_size=self.hidden_size, num_heads=self.num_heads, **kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward pass through sequential Mamba and Attention blocks with residuals.

        Args:
            hidden_states: Input tensor of shape [B, L, D].
            attention_mask: Optional attention mask.

        Returns:
            Output tensor of shape [B, L, D].
        """
        # Mamba pass
        mamba_out = self.mamba(self.mamba_norm(hidden_states), **kwargs)
        m_tensor = mamba_out[0] if isinstance(mamba_out, tuple) else mamba_out
        x = hidden_states + m_tensor

        # Attention pass
        attn_out = self.attn(self.attn_norm(x), attention_mask=attention_mask, **kwargs)
        a_tensor = attn_out[0] if isinstance(attn_out, tuple) else attn_out
        return x + a_tensor


__all__ = ["ZambaBlock"]
