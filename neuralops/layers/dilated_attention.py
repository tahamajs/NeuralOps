#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Dilated Attention Layer.

Dilated sliding attention (Ding et al., LongNet 2023) expanding receptive field
reach over exponential sequence scales via structured strided subsampling.
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

logger = logging.getLogger(__name__)


@ATTENTION_REGISTRY.register(
    "dilated_attention",
    supports_causal=True,
    supports_bidirectional=True,
    supports_gqa=True,
)
class DilatedAttention(BaseAttention):
    """Dilated attention operator with strided token interaction patterns.

    Attributes:
        supports_causal: Whether causal dilated masking is supported.
        supports_bidirectional: Whether bidirectional dilated attention is supported.
        supports_gqa: Whether grouped-query head broadcasting is supported.
    """

    FAMILY: str = "sparse"
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
        dilation_rate: int = 2,
        segment_size: int = 64,
        dropout: float = 0.0,
        bias: bool = True,
        is_decoder: bool = False,
        **kwargs: Any,  # reason: arbitrary hyperparameters
    ) -> None:
        """Initializes DilatedAttention."""
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
            or _cfg_get(config, "num_heads", 8)
        )
        self.num_heads: int = max(1, int(resolved_heads))

        resolved_head_dim = head_dim or (self.dim // self.num_heads)
        self.head_dim: int = int(resolved_head_dim)
        self.all_head_size: int = self.num_heads * self.head_dim

        self.dilation_rate: int = max(1, int(dilation_rate or _cfg_get(config, "dilation_rate", 2)))
        self.segment_size: int = max(1, int(segment_size or _cfg_get(config, "segment_size", 64)))
        self.is_decoder: bool = bool(is_decoder or _cfg_get(config, "is_decoder", False))

        self.q_proj = LoRALinear(self.dim, self.all_head_size, bias=bias, freeze_base=True)
        self.k_proj = LoRALinear(self.dim, self.all_head_size, bias=bias, freeze_base=True)
        self.v_proj = LoRALinear(self.dim, self.all_head_size, bias=bias, freeze_base=True)
        self.out_proj = LoRALinear(self.all_head_size, self.dim, bias=bias, freeze_base=True)

        dropout_p = float(dropout or _cfg_get(config, "dropout", 0.0))
        self.dropout = nn.Dropout(dropout_p)
        self.scale: float = 1.0 / math.sqrt(max(self.head_dim, 1))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *args: Any,  # reason: absorbs unused positional args
        **kwargs: Any,  # reason: absorbs unused kwargs
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass with strided dilated token gathering and local attention."""
        b, seq_len, _ = hidden_states.shape
        h, d = self.num_heads, self.head_dim
        r = self.dilation_rate

        q = self.q_proj(hidden_states).view(b, seq_len, h, d).transpose(1, 2)
        k = self.k_proj(hidden_states).view(b, seq_len, h, d).transpose(1, 2)
        v = self.v_proj(hidden_states).view(b, seq_len, h, d).transpose(1, 2)

        # For small sequences where seq_len <= dilation_rate, fallback to full attention
        if seq_len <= r:
            scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * self.scale
            if self.is_decoder and seq_len > 1:
                mask = torch.triu(
                    torch.ones((seq_len, seq_len), device=hidden_states.device, dtype=torch.bool), diagonal=1
                )
                scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), -1e4)
            weights = F.softmax(scores, dim=-1).to(hidden_states.dtype)
            weights = self.dropout(torch.nan_to_num(weights, nan=0.0))
            context = torch.matmul(weights, v).transpose(1, 2).contiguous().view(b, seq_len, self.all_head_size)
            return self.out_proj(context), weights

        # Strided dilated subsampling: gather every r-th token across offset phases
        out_context = torch.zeros_like(q)
        for offset in range(r):
            idx = torch.arange(offset, seq_len, r, device=hidden_states.device)
            if idx.numel() == 0:
                continue
            q_sub = q.index_select(-2, idx)
            k_sub = k.index_select(-2, idx)
            v_sub = v.index_select(-2, idx)
            l_sub = idx.numel()

            scores_sub = torch.matmul(q_sub.float(), k_sub.float().transpose(-2, -1)) * self.scale
            if self.is_decoder and l_sub > 1:
                causal_sub = torch.triu(
                    torch.ones((l_sub, l_sub), device=hidden_states.device, dtype=torch.bool), diagonal=1
                )
                scores_sub = scores_sub.masked_fill(causal_sub.unsqueeze(0).unsqueeze(0), -1e4)

            w_sub = F.softmax(scores_sub, dim=-1).to(hidden_states.dtype)
            w_sub = self.dropout(torch.nan_to_num(w_sub, nan=0.0))
            c_sub = torch.matmul(w_sub, v_sub)

            out_context.index_copy_(-2, idx, c_sub)

        context = out_context.transpose(1, 2).contiguous().view(b, seq_len, self.all_head_size)
        out = self.out_proj(context)
        return out, None


dilated_attention = DilatedAttention

__all__ = [
    "DilatedAttention",
    "dilated_attention",
]
