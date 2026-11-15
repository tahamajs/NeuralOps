#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Perceiver Resampler Module.

Flamingo-style visual/multimodal Perceiver Resampler that compresses
arbitrary-length context sequences into a fixed number of latent tokens
via interleaved cross-attention and feed-forward layers.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.base import BaseAttention
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.layers.perceiver import PerceiverAttention

logger = logging.getLogger(__name__)


@ATTENTION_REGISTRY.register("perceiver_resampler")
class PerceiverResampler(BaseAttention):
    """Perceiver Resampler compressing arbitrary input sequences to fixed latent length.

    Args:
        config: Model configuration object.
        dim: Representation dimension.
        num_latents: Number of output tokens (default: 64).
        num_heads: Number of attention heads.
        depth: Number of resampler layers (default: 2).
    """

    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,
        dim: int = 512,
        num_latents: int = 64,
        num_heads: int = 8,
        depth: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, **kwargs)
        self.dim = int(getattr(config, "hidden_size", dim))
        self.num_latents = int(num_latents)
        self.depth = int(depth)

        self.cross_attn = PerceiverAttention(
            config=config,
            dim=self.dim,
            num_heads=num_heads,
            num_latents=self.num_latents,
            **kwargs,
        )
        self.ln_latents = nn.LayerNorm(self.dim)
        self.ln_context = nn.LayerNorm(self.dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim, self.dim * 4),
            nn.GELU(),
            nn.Linear(self.dim * 4, self.dim),
        )
        self.post_norm = nn.LayerNorm(self.dim)

    def forward(
        self,
        context: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Resamples context into fixed num_latents representation.

        Args:
            context: Context tensor [batch_size, seq_len, dim].
            attention_mask: Optional attention mask over context.

        Returns:
            Resampled output tensor [batch_size, num_latents, dim].
        """
        normed_context = self.ln_context(context)
        latents = self.cross_attn(normed_context, attention_mask=attention_mask, **kwargs)
        latents = latents + self.mlp(self.ln_latents(latents))
        return self.post_norm(latents)


__all__ = ["PerceiverResampler"]
