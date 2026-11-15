#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""FlashAttention Layer.

Fast, exact, memory-efficient attention (Dao et al., 2022) with tiling across SRAM
and graceful device fallback for CPU, MPS, and arbitrary precisions.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.attention_utils import LoRALinear
from neuralops.core.base import BaseAttention
from neuralops.core.common import _cfg_get
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.layers.sdpa import SDPABackend, sdpa_forward

logger = logging.getLogger(__name__)


class FlashAttention(BaseAttention):
    """FlashAttention neural operator with hardware-adaptive execution.

    Attributes:
        supports_causal: True.
        supports_bidirectional: True.
        supports_gqa: True.
    """

    FAMILY: str = "fast_kernels"
    supports_causal: bool = True
    supports_bidirectional: bool = True
    supports_gqa: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,  # reason: HuggingFace config or dict
        dim: int | None = None,
        num_heads: int | None = None,
        head_dim: int | None = None,
        dropout: float = 0.0,
        bias: bool = True,
        is_decoder: bool = False,
        **kwargs: Any,  # reason: arbitrary hyperparameters
    ) -> None:
        """Initializes FlashAttention."""
        super().__init__(config=config)

        resolved_dim = (
            dim
            or kwargs.get("hidden_dim")
            or kwargs.get("hidden_size")
            or kwargs.get("d_model")
            or _cfg_get(config, "hidden_dim", None)
            or _cfg_get(config, "hidden_size", None)
            or _cfg_get(config, "dim", 768)
        )
        self.dim: int = int(resolved_dim)
        self.hidden_size: int = self.dim

        resolved_heads = (
            num_heads
            or kwargs.get("num_attention_heads")
            or _cfg_get(config, "num_attention_heads", None)
            or _cfg_get(config, "num_heads", 12)
        )
        self.num_heads: int = max(1, int(resolved_heads))

        resolved_head_dim = head_dim or (self.dim // self.num_heads)
        self.head_dim: int = int(resolved_head_dim)
        self.all_head_size: int = self.num_heads * self.head_dim
        self.is_decoder: bool = bool(is_decoder or _cfg_get(config, "is_decoder", False))

        self.q_proj = LoRALinear(self.dim, self.all_head_size, bias=bias, freeze_base=True)
        self.k_proj = LoRALinear(self.dim, self.all_head_size, bias=bias, freeze_base=True)
        self.v_proj = LoRALinear(self.dim, self.all_head_size, bias=bias, freeze_base=True)
        self.out_proj = LoRALinear(self.all_head_size, self.dim, bias=bias, freeze_base=True)

        dropout_p = float(dropout or _cfg_get(config, "dropout", 0.0))
        self.dropout_p: float = dropout_p

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *args: Any,  # reason: absorbs unused positional args
        **kwargs: Any,  # reason: absorbs unused kwargs
    ) -> tuple[torch.Tensor, None]:
        """Forward pass executing FlashAttention with automatic hardware fallback."""
        b, seq_len, _ = hidden_states.shape
        h, d = self.num_heads, self.head_dim

        q = self.q_proj(hidden_states).view(b, seq_len, h, d).transpose(1, 2)
        k = self.k_proj(hidden_states).view(b, seq_len, h, d).transpose(1, 2)
        v = self.v_proj(hidden_states).view(b, seq_len, h, d).transpose(1, 2)

        is_causal = self.is_decoder and seq_len > 1
        dropout_p = self.dropout_p if self.training else 0.0

        attn_mask = None
        if attention_mask is not None:
            if attention_mask.dim() == 4:
                attn_mask = attention_mask.to(q.dtype)
            elif attention_mask.dim() == 2:
                attn_mask = ((attention_mask[:, None, None, :].float() - 1.0) * 1e4).to(q.dtype)

        # Execute via unified SDPA dispatcher which targets FLASH on eligible hardware
        context = sdpa_forward(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal and attn_mask is None,
            backend=SDPABackend.AUTO,
        )

        context = context.transpose(1, 2).contiguous().view(b, seq_len, self.all_head_size)
        out = self.out_proj(context)
        return out, None


FlashAttentionLayer = FlashAttention
flash_attention = FlashAttention

__all__ = [
    "FlashAttention",
    "FlashAttentionLayer",
    "flash_attention",
]
