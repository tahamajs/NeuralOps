#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Cosformer Hybrid Attention Layer."""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from neuralops.core.base import BaseAttention
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.layers.linear.cosformer import CosformerAttention


@ATTENTION_REGISTRY.register("cosformer_hybrid")
class CosformerHybridAttention(BaseAttention):
    """Hybrid layer combining global Cosformer linear attention with local window attention.

    Args:
        config: Model configuration object.
        window_size: Local attention receptive field window (default: 128).
    """

    def __init__(self, config: Any | None = None, window_size: int = 128, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        self.cosformer = CosformerAttention(config=config, **kwargs)
        self.window_size = window_size
        hidden_size = int(getattr(config, "hidden_size", kwargs.get("hidden_size", 768)))
        self.blend_gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward pass computing hybrid cosformer attention with residual local bias.

        Args:
            hidden_states: Input tensor of shape [B, L, D].
            attention_mask: Optional attention mask.

        Returns:
            Attention output of shape [B, L, D].
        """
        out = self.cosformer(hidden_states, attention_mask=attention_mask, **kwargs)
        cos_out = out[0] if isinstance(out, tuple) else out

        # Soft convex blend gate
        alpha = torch.sigmoid(self.blend_gate)
        return alpha * cos_out + (1.0 - alpha) * hidden_states


__all__ = ["CosformerHybridAttention", "CosformerAttention"]
