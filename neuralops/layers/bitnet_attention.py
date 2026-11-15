#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
BitNet b1.58 Ternary Attention Layer.

Quantizes linear projection weights into ternary {-1, 0, 1} states with
activation scaling and Straight-Through Estimator (STE) gradient propagation.
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


def weight_quant_ste(w: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Quantizes weights to ternary values {-1, 0, 1} via STE absmean.

    Args:
        w: Weight tensor of shape [out_features, in_features].
        eps: Small epsilon preventing division by zero.

    Returns:
        Quantized weight tensor with straight-through gradient copy.
    """
    scale = w.abs().mean().clamp(min=eps)
    w_scaled = w / scale
    w_q = torch.clamp(torch.round(w_scaled), -1.0, 1.0) * scale
    return w + (w_q - w).detach()


def activation_quant_ste(x: torch.Tensor, num_bits: int = 8, eps: float = 1e-8) -> torch.Tensor:
    """Quantizes activations to 8-bit integers via per-token max scaling.

    Args:
        x: Input activation tensor.
        num_bits: Bit-width for activation quantization (default: 8).
        eps: Small epsilon preventing division by zero.

    Returns:
        Quantized activation tensor with straight-through gradient copy.
    """
    qmax = float((1 << (num_bits - 1)) - 1)
    scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=eps) / qmax
    x_q = torch.clamp(torch.round(x / scale), -qmax, qmax) * scale
    return x + (x_q - x).detach()


class BitLinear(nn.Linear):
    """Linear layer using ternary weights and 8-bit activations."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        w_quant = weight_quant_ste(self.weight)
        x_quant = activation_quant_ste(input)
        return F.linear(x_quant, w_quant, self.bias)


@ATTENTION_REGISTRY.register("bitnet_attention")
class BitNetAttention(BaseAttention):
    """Attention layer with BitNet b1.58 ternary linear projections.

    Args:
        config: Model configuration object.
        hidden_size: Hidden representation size.
        num_heads: Number of attention heads.
    """

    supports_causal: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,
        hidden_size: int = 768,
        num_heads: int = 12,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, **kwargs)
        self.hidden_size = int(getattr(config, "hidden_size", hidden_size))
        self.num_heads = int(getattr(config, "num_attention_heads", num_heads))
        self.head_dim = self.hidden_size // max(1, self.num_heads)

        self.q_proj = BitLinear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = BitLinear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = BitLinear(self.hidden_size, self.hidden_size, bias=False)
        self.out_proj = BitLinear(self.hidden_size, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward pass with ternary projections and scaled dot-product attention.

        Args:
            hidden_states: Input tensor of shape [B, L, D].
            attention_mask: Optional attention mask.

        Returns:
            Output tensor of shape [B, L, D].
        """
        B, L, D = hidden_states.shape
        q = self.q_proj(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        is_causal = kwargs.get("is_causal", kwargs.get("causal", False))
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            is_causal=is_causal,
        )
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out)


__all__ = [
    "BitNetAttention",
    "BitLinear",
    "weight_quant_ste",
    "activation_quant_ste",
]
