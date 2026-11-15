#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Perceiver: Latent Bottleneck Cross-Attention Architecture.

Implements DeepMind's Perceiver (Jaegle et al.) cross-attention mechanism:
    Input sequence of length T is queried by M learned latent vectors (M << T),
    yielding an O(M * T) bottleneck representation that decouples model depth
    from input sequence length.
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


@ATTENTION_REGISTRY.register("perceiver")
class PerceiverAttention(BaseAttention):
    """Perceiver Cross-Attention Module.

    Queries context input representations using learned or passed latent queries.

    Args:
        config: Model configuration object.
        dim: Latent dimension size.
        dim_context: Input context sequence dimension size.
        num_heads: Number of attention heads.
        num_latents: Number of learnable latent tokens (default: 64).
    """

    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,
        dim: int = 512,
        dim_context: Optional[int] = None,
        num_heads: int = 8,
        num_latents: int = 64,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, **kwargs)
        self.dim = int(getattr(config, "hidden_size", dim))
        self.dim_context = int(dim_context if dim_context is not None else self.dim)
        self.num_heads = int(getattr(config, "num_attention_heads", num_heads))
        self.head_dim = self.dim // max(1, self.num_heads)
        self.num_latents = int(num_latents)

        # Learnable latent queries
        self.latents = nn.Parameter(torch.randn(self.num_latents, self.dim) * 0.02)

        self.q_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.k_proj = nn.Linear(self.dim_context, self.dim, bias=False)
        self.v_proj = nn.Linear(self.dim_context, self.dim, bias=False)
        self.out_proj = nn.Linear(self.dim, self.dim, bias=False)

    def forward(
        self,
        context: torch.Tensor,
        latents: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Cross-attends latents over context sequence.

        Args:
            context: Context tensor [batch_size, seq_len, dim_context].
            latents: Optional query latents override [batch_size, num_latents, dim].
            attention_mask: Optional attention mask over context.

        Returns:
            Updated latent representations [batch_size, num_latents, dim].
        """
        B = context.shape[0]
        if latents is None:
            # Broadcast learned latents over batch
            latents = self.latents.unsqueeze(0).expand(B, -1, -1)

        M = latents.shape[1]
        T = context.shape[1]

        q = self.q_proj(latents).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(context).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(context).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            is_causal=False,
        )
        out = out.transpose(1, 2).contiguous().view(B, M, self.dim)
        return self.out_proj(out)


__all__ = ["PerceiverAttention"]
