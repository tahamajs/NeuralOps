#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Native Multi-Head Attention Layer.

Pure PyTorch multi-head attention avoiding hardware-specific accelerator bindings
to provide deterministic cross-platform execution on CPU, CUDA, MPS, and ROCm.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.attention_mask import get_token_valid_mask
from neuralops.core.base import BaseAttention
from neuralops.core.common import _cfg_get
from neuralops.core.registry import ATTENTION_REGISTRY

__all__ = [
    "NativeAttention",
]


@ATTENTION_REGISTRY.register(
    "native_attention",
    supports_causal=True,
    supports_bidirectional=True,
    supports_gqa=True,
)
class NativeAttention(BaseAttention):
    """Pure PyTorch native multi-head attention module.

    Attributes:
        supports_causal: Whether causal autoregressive masking is supported.
        supports_bidirectional: Whether bidirectional attention is supported.
        supports_gqa: Whether grouped-query head broadcasting is supported.
    """

    FAMILY: str = "dense"
    supports_causal: bool = True
    supports_bidirectional: bool = True
    supports_gqa: bool = True
    is_custom_attention: bool = False

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
        """Initializes NativeAttention."""
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

        self.q_proj = nn.Linear(self.dim, self.all_head_size, bias=bias)
        self.k_proj = nn.Linear(self.dim, self.all_head_size, bias=bias)
        self.v_proj = nn.Linear(self.dim, self.all_head_size, bias=bias)
        self.out_proj = nn.Linear(self.all_head_size, self.dim, bias=bias)

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
        """Forward pass using pure PyTorch tensor operations."""
        b, seq_len, _ = hidden_states.shape
        h, d = self.num_heads, self.head_dim

        q = self.q_proj(hidden_states).view(b, seq_len, h, d).transpose(1, 2)
        k = self.k_proj(hidden_states).view(b, seq_len, h, d).transpose(1, 2)
        v = self.v_proj(hidden_states).view(b, seq_len, h, d).transpose(1, 2)

        scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * self.scale

        if self.is_decoder and seq_len > 1:
            causal_mask = torch.triu(
                torch.ones((seq_len, seq_len), device=hidden_states.device, dtype=torch.bool), diagonal=1
            )
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), -1e4)

        if attention_mask is not None:
            if attention_mask.dim() == 4:
                scores = scores + attention_mask.to(scores.dtype)
            else:
                valid_m = get_token_valid_mask(attention_mask, b, seq_len, hidden_states.device)
                if valid_m is not None:
                    scores = scores.masked_fill(~valid_m[:, None, None, :], -1e4)

        weights = F.softmax(scores, dim=-1).to(hidden_states.dtype)
        weights = torch.nan_to_num(weights, nan=0.0)
        weights = self.dropout(weights)

        context = torch.matmul(weights, v).transpose(1, 2).contiguous().view(b, seq_len, self.all_head_size)
        out = self.out_proj(context)
        return out, weights
