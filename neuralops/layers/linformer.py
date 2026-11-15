#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Linformer linear attention layer with low-rank sequence projection."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.base import BaseAttention
from neuralops.core.registry import ATTENTION_REGISTRY


class LinformerAttention(BaseAttention):
    """Linformer Attention with low-rank projected sequence length."""

    supports_causal: bool = False
    supports_bidirectional: bool = True
    supports_gqa: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,
        k: int = 64,
        max_seq_len: int = 512,
        dim: int | None = None,
        num_heads: int = 4,
        dropout: float = 0.0,
        **kwargs: Any,
    ) -> None:
        """Initialize LinformerAttention.

        Args:
            config: Optional config.
            k: Projected sequence length.
            max_seq_len: Maximum input sequence length.
            dim: Embedding dimension.
            num_heads: Number of attention heads.
            dropout: Attention dropout.
            **kwargs: Extra parameters.
        """
        super().__init__(config=config if not isinstance(config, int) else None, **kwargs)

        resolved_dim = (
            dim
            or kwargs.get("hidden_size")
            or kwargs.get("d_model")
            or (getattr(config, "hidden_size", getattr(config, "dim", None)) if config is not None else None)
            or 64
        )
        self.dim = int(resolved_dim)
        self.hidden_size = self.dim

        resolved_heads = (
            kwargs.get("num_heads")
            or kwargs.get("num_attention_heads")
            or (
                getattr(config, "num_attention_heads", getattr(config, "num_heads", None))
                if config is not None
                else None
            )
            or num_heads
        )
        self.num_heads = max(1, int(resolved_heads))
        self.head_dim = max(1, self.dim // self.num_heads)

        self.k = max(1, int(kwargs.get("k", k)))
        self.max_seq_len = max(self.k, int(kwargs.get("max_seq_len", max_seq_len)))

        self.q_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.k_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.v_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.out_proj = nn.Linear(self.dim, self.dim, bias=False)

        self.e_proj = nn.Linear(self.max_seq_len, self.k, bias=False)
        self.f_proj = nn.Linear(self.max_seq_len, self.k, bias=False)
        self.dropout = nn.Dropout(float(dropout))

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        """Forward pass for LinformerAttention.

        Args:
            x: Input tensor [B, N, D].
            attention_mask: Optional padding mask.
            **kwargs: Extra parameters.

        Returns:
            Tuple of (output, None).
        """
        B, N, D = x.shape
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Pad or slice to max_seq_len for projection
        if N < self.max_seq_len:
            k_padded = F.pad(k, (0, 0, 0, self.max_seq_len - N))
            v_padded = F.pad(v, (0, 0, 0, self.max_seq_len - N))
        else:
            k_padded = k[:, :, : self.max_seq_len, :]
            v_padded = v[:, :, : self.max_seq_len, :]

        # Project key and value sequence length from N to k
        k_proj = self.e_proj(k_padded.transpose(-2, -1)).transpose(-2, -1)  # [B, H, k, head_dim]
        v_proj = self.f_proj(v_padded.transpose(-2, -1)).transpose(-2, -1)  # [B, H, k, head_dim]

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k_proj.transpose(-2, -1)) * scale  # [B, H, N, k]

        weights = F.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        weights = self.dropout(weights)

        context = torch.matmul(weights, v_proj)  # [B, H, N, head_dim]
        out = context.transpose(1, 2).contiguous().view(B, N, D)
        out = self.out_proj(out)
        return out, None


try:
    ATTENTION_REGISTRY._dict["linformer_attention"] = LinformerAttention
    ATTENTION_REGISTRY._aliases["linformer"] = LinformerAttention
except Exception:
    pass

__all__ = ["LinformerAttention"]
