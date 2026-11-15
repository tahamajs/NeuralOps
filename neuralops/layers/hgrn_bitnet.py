#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
HGRN-BitNet: Hierarchically Gated Linear Recurrence with Ternary Weights.

Fuses HGRN's sub-quadratic gated linear recurrence with BitNet b1.58's
ternary weight quantization {-1, 0, 1} for ultra-low-memory inference.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.base import BaseAttention
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.layers.bitnet_attention import BitLinear

logger = logging.getLogger(__name__)


@ATTENTION_REGISTRY.register("hgrn_bitnet")
class HGRNBitNetAttention(BaseAttention):
    """HGRN linear recurrence with ternary BitNet projections.

    Args:
        config: Model configuration object.
        hidden_size: Hidden representation dimension.
    """

    supports_causal: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,
        hidden_size: int = 768,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, **kwargs)
        self.hidden_size = int(getattr(config, "hidden_size", hidden_size))

        self.in_proj = BitLinear(self.hidden_size, self.hidden_size, bias=False)
        self.gate_proj = BitLinear(self.hidden_size, self.hidden_size, bias=True)
        self.out_proj = BitLinear(self.hidden_size, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward pass with ternary projections and gated cumulative recurrence.

        Args:
            hidden_states: Input tensor of shape [B, L, D].
            attention_mask: Optional attention mask.

        Returns:
            Output tensor of shape [B, L, D].
        """
        B, L, D = hidden_states.shape
        x = self.in_proj(hidden_states)
        g = torch.sigmoid(self.gate_proj(hidden_states).float())

        # Gated cumulative recurrence in float32 for numerical stability
        h = torch.zeros(B, D, device=hidden_states.device, dtype=torch.float32)
        outputs = []
        for t in range(L):
            h = g[:, t] * h + (1.0 - g[:, t]) * x[:, t].float()
            outputs.append(h.to(hidden_states.dtype))

        out_stack = torch.stack(outputs, dim=1)
        return self.out_proj(out_stack)


__all__ = ["HGRNBitNetAttention"]
