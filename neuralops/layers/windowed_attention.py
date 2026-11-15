#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Windowed (Sliding Window) Attention Layer.

Restricts attention interactions to a fixed local receptive field window of radius W,
achieving O(N * W) compute and memory complexity while retaining linear scalability.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.attention_mask import get_token_valid_mask
from neuralops.core.attention_utils import LoRALinear
from neuralops.core.base import BaseAttention
from neuralops.core.common import _cfg_get
from neuralops.core.registry import ATTENTION_REGISTRY

logger = logging.getLogger(__name__)


@ATTENTION_REGISTRY.register(
    "windowed_attention",
    supports_causal=True,
    supports_bidirectional=True,
    supports_gqa=True,
)
class WindowedAttention(BaseAttention):
    """Sliding window local attention operator.

    Attributes:
        supports_causal: Whether causal sliding window masking is supported.
        supports_bidirectional: Whether bidirectional local window masking is supported.
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
        window_size: int = 128,
        dropout: float = 0.0,
        bias: bool = True,
        is_decoder: bool = False,
        **kwargs: Any,  # reason: arbitrary hyperparameters
    ) -> None:
        """Initializes WindowedAttention."""
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
        self.window_size: int = int(window_size or _cfg_get(config, "sliding_window", 128))
        self.is_decoder: bool = bool(is_decoder or _cfg_get(config, "is_decoder", False))

        self.q_proj = LoRALinear(self.dim, self.all_head_size, bias=bias, freeze_base=True)
        self.k_proj = LoRALinear(self.dim, self.all_head_size, bias=bias, freeze_base=True)
        self.v_proj = LoRALinear(self.dim, self.all_head_size, bias=bias, freeze_base=True)
        self.out_proj = LoRALinear(self.all_head_size, self.dim, bias=bias, freeze_base=True)

        dropout_p = float(dropout or _cfg_get(config, "dropout", 0.0))
        self.dropout = nn.Dropout(dropout_p)
        self.scale: float = 1.0 / math.sqrt(max(self.head_dim, 1))

    def _build_window_mask(self, seq_q: int, seq_k: int, device: torch.device) -> torch.Tensor:
        """Constructs boolean mask where True indicates allowed local attention positions."""
        q_idx = torch.arange(seq_q, device=device).unsqueeze(1) + (seq_k - seq_q)
        k_idx = torch.arange(seq_k, device=device).unsqueeze(0)
        diff = q_idx - k_idx

        if self.is_decoder:
            # Causal window: 0 <= diff <= window_size
            return (diff >= 0) & (diff <= self.window_size)
        # Bidirectional window: |diff| <= window_size
        return diff.abs() <= self.window_size

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *args: Any,  # reason: absorbs unused positional args
        **kwargs: Any,  # reason: absorbs unused kwargs
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward sliding window attention pass."""
        b, seq_len, _ = hidden_states.shape
        h, d = self.num_heads, self.head_dim

        q = self.q_proj(hidden_states).view(b, seq_len, h, d).transpose(1, 2)
        k = self.k_proj(hidden_states).view(b, seq_len, h, d).transpose(1, 2)
        v = self.v_proj(hidden_states).view(b, seq_len, h, d).transpose(1, 2)

        scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * self.scale
        window_mask = self._build_window_mask(seq_len, seq_len, hidden_states.device)
        scores = scores.masked_fill(~window_mask.unsqueeze(0).unsqueeze(0), -1e4)

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


windowed_attention = WindowedAttention

__all__ = [
    "WindowedAttention",
    "windowed_attention",
]
