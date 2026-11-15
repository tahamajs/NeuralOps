#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Transformer-Mamba Hybrid Attention Layer.

Integrates multi-head self-attention with Mamba selective state space recurrence
to balance long-sequence induction head retention and linear-time sequence processing.
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


@ATTENTION_REGISTRY.register("transformer_mamba")
class TransformerMambaBlock(BaseAttention):
    """Hybrid block interweaving Multi-Head Attention and Mamba SSM.

    Args:
        config: Model configuration object.
        hidden_size: Hidden representation dimension.
        num_heads: Number of attention heads.
        mamba_d_state: State expansion rank for SSM.
    """

    def __init__(
        self,
        config: Any | None = None,
        hidden_size: int = 768,
        num_heads: int = 12,
        mamba_d_state: int = 16,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, **kwargs)
        h_size = int(getattr(config, "hidden_size", hidden_size))
        n_heads = int(getattr(config, "num_attention_heads", num_heads))

        self.attn_norm = nn.LayerNorm(h_size)
        self.attn = FullAttention(config=config, hidden_size=h_size, num_heads=n_heads, **kwargs)

        self.ssm_norm = nn.LayerNorm(h_size)
        self.ssm = MambaBlock(config=config, hidden_size=h_size, d_state=mamba_d_state, **kwargs)

        # Learnable blending factor between SSM and Attention
        self.hybrid_gate = nn.Parameter(torch.zeros(h_size))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward pass through hybrid Mamba-Attention block.

        Args:
            hidden_states: Tensor of shape [B, L, D].
            attention_mask: Optional attention mask.

        Returns:
            Output tensor of shape [B, L, D].
        """
        # Attention branch with residual
        normed_attn = self.attn_norm(hidden_states)
        attn_out = self.attn(normed_attn, attention_mask=attention_mask, **kwargs)
        attn_tensor = attn_out[0] if isinstance(attn_out, tuple) else attn_out

        # SSM branch
        normed_ssm = self.ssm_norm(hidden_states)
        ssm_out = self.ssm(normed_ssm, **kwargs)
        ssm_tensor = ssm_out[0] if isinstance(ssm_out, tuple) else ssm_out

        # Gated fusion with residual connection
        gate = torch.sigmoid(self.hybrid_gate)
        blended = gate * attn_tensor + (1.0 - gate) * ssm_tensor
        return hidden_states + blended


__all__ = ["TransformerMambaBlock"]
