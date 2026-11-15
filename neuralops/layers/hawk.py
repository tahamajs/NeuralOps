#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Hawk: Pure Gated Linear Recurrence Architecture.

Implements the Real-Gated Linear Recurrent Unit (RG-LRU) from DeepMind's Griffin/Hawk:
    a_t = sigmoid(W_a x_t + b_a)
    h_t = a_t * h_{t-1} + sqrt(1 - a_t^2) * (W_x x_t)
    y_t = W_out (h_t * gelu(W_g x_t))
Provides linear-time training and O(1) step inference.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.base import BaseAttention
from neuralops.core.registry import ATTENTION_REGISTRY

logger = logging.getLogger(__name__)


@ATTENTION_REGISTRY.register("hawk")
class HawkAttention(BaseAttention):
    """Hawk Real-Gated Linear Recurrence Layer (RG-LRU).

    Args:
        config: Model configuration object.
        hidden_size: Model embedding dimension.
        expand_factor: Internal state expansion ratio (default: 1.5).
    """

    supports_causal: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,
        hidden_size: int = 768,
        expand_factor: float = 1.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, **kwargs)
        self.hidden_size = int(getattr(config, "hidden_size", hidden_size))
        self.d_inner = int(self.hidden_size * expand_factor)

        # Projections
        self.in_proj = nn.Linear(self.hidden_size, self.d_inner, bias=False)
        self.gate_proj = nn.Linear(self.hidden_size, self.d_inner, bias=False)
        self.alpha_proj = nn.Linear(self.hidden_size, self.d_inner, bias=True)

        # Learnable decay gate bias initialization: high retention (e.g. log(8.0))
        nn.init.constant_(self.alpha_proj.bias, 2.0)

        self.out_proj = nn.Linear(self.d_inner, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_state: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]] | torch.Tensor:
        """Forward pass through Hawk RG-LRU.

        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size].
            attention_mask: Optional mask tensor.
            past_state: Optional prior recurrent state tensor [batch_size, d_inner].
            use_cache: Whether to return updated recurrent state.

        Returns:
            Output tensor [batch_size, seq_len, hidden_size] or tuple with state.
        """
        batch_size, seq_len, _ = hidden_states.shape
        orig_dtype = hidden_states.dtype

        # Compute branch representations
        x = self.in_proj(hidden_states)
        gate = F.gelu(self.gate_proj(hidden_states))
        a = torch.sigmoid(self.alpha_proj(hidden_states).float())  # compute retention in float32

        # RG-LRU input scaling sqrt(1 - a^2)
        x_scaled = (torch.sqrt(torch.clamp(1.0 - a**2, min=1e-7)) * x.float()).to(orig_dtype)

        # Recurrent scan in float32 for numerical stability
        h = (
            past_state.float()
            if past_state is not None
            else torch.zeros(batch_size, self.d_inner, device=hidden_states.device)
        )
        outputs = []
        for t in range(seq_len):
            h = a[:, t] * h + x_scaled[:, t].float()
            y_t = h.to(orig_dtype) * gate[:, t]
            outputs.append(y_t)

        out_stack = torch.stack(outputs, dim=1)
        final_out = self.out_proj(out_stack)

        if use_cache or kwargs.get("return_state", False):
            return final_out, h.to(orig_dtype)
        return final_out


__all__ = ["HawkAttention"]
