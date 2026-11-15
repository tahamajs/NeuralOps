#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
RecurrentGemma Layer Architecture.

Combines Griffin-style Real-Gated Linear Recurrence (RG-LRU) with Gemma's RMSNorm
and local sliding-window attention mechanics for efficient O(1) step inference.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.base import BaseAttention
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.layers.hawk import HawkAttention

logger = logging.getLogger(__name__)


class GemmaRMSNorm(nn.Module):
    """RMSNorm with Gemma offset convention: (1 + weight) * rrms(x)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        normed = x * torch.rsqrt(variance + self.eps)
        return normed * (1.0 + self.weight)


@ATTENTION_REGISTRY.register("recurrent_gemma")
class RecurrentGemmaAttention(BaseAttention):
    """RecurrentGemma hybrid recurrent block with RMSNorm.

    Args:
        config: Model configuration object.
        hidden_size: Hidden size dimension (e.g. 2048 for Gemma 2B).
        num_heads: Attention / recurrence head count.
    """

    supports_causal: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,
        hidden_size: int = 768,
        num_heads: int = 8,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, **kwargs)
        self.hidden_size = int(getattr(config, "hidden_size", hidden_size))
        self.num_heads = int(getattr(config, "num_attention_heads", num_heads))

        self.input_layernorm = GemmaRMSNorm(self.hidden_size)
        self.recurrent_layer = HawkAttention(config=config, hidden_size=self.hidden_size, **kwargs)
        self.post_attention_layernorm = GemmaRMSNorm(self.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_state: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]] | torch.Tensor:
        """Forward pass applying RMSNorm -> Hawk RG-LRU -> residual connection.

        Args:
            hidden_states: Input tensor of shape [B, L, D].
            attention_mask: Optional attention mask.
            past_state: Optional past recurrent state.
            use_cache: Whether to return updated recurrent state.

        Returns:
            Output tensor [B, L, D] or (output, state) tuple.
        """
        normed = self.input_layernorm(hidden_states)
        out = self.recurrent_layer(
            normed,
            attention_mask=attention_mask,
            past_state=past_state,
            use_cache=use_cache,
            **kwargs,
        )
        if isinstance(out, tuple):
            rec_out, new_state = out
            return self.post_attention_layernorm(rec_out) + hidden_states, new_state

        return self.post_attention_layernorm(out) + hidden_states


__all__ = ["RecurrentGemmaAttention", "GemmaRMSNorm"]
