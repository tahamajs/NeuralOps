#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""MoBA Block Router module for mixture-of-block-attention routing."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.ops.triton.block_sparse import (
    fused_moba_block_router_combine_pytorch as _fused_moba_combine,
)

fused_moba_block_router_combine_pytorch = _fused_moba_combine


class MoBABlockRouter(nn.Module):
    """Learned Block Router for Mixture of Block Attention (MoBA).

    Deterministically scores and selects top-k key-value blocks for each query block.
    """

    def __init__(
        self,
        d_model: int = 64,
        num_blocks: int = 8,
        router_hidden: int = 32,
        top_k: int = 2,
    ) -> None:
        """Initialize MoBABlockRouter.

        Args:
            d_model: Feature dimension.
            num_blocks: Number of blocks in the sequence.
            router_hidden: Hidden dimension of router MLP.
            top_k: Number of blocks to select.
        """
        super().__init__()
        self.d_model = int(d_model)
        self.num_blocks = max(1, int(num_blocks))
        self.top_k = max(1, int(top_k))
        self.net = nn.Sequential(
            nn.Linear(self.d_model, router_hidden),
            nn.GELU(),
            nn.Linear(router_hidden, self.num_blocks),
        )

    def forward(
        self,
        query_blocks: torch.Tensor,
        top_k: int | None = None,
        is_causal: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute block routing logits and deterministic top-k selected blocks.

        Args:
            query_blocks: Tensor of shape [B, num_blocks, d_model] or [B, d_model].
            top_k: Optional override for k.
            is_causal: If True, mask out future blocks.

        Returns:
            Tuple of (logits [B, num_blocks, num_blocks], topk_indices [B, num_blocks, k]).
        """
        if query_blocks.ndim == 2:
            query_blocks = query_blocks.unsqueeze(1)

        B, nb, d = query_blocks.shape
        logits = self.net(query_blocks)  # [B, nb, num_blocks]

        if is_causal:
            idx = torch.arange(nb, device=query_blocks.device)
            causal_mask = idx.unsqueeze(1) >= idx.unsqueeze(0)
            logits = logits.masked_fill(~causal_mask.unsqueeze(0), float("-inf"))

        k_val = min(top_k or self.top_k, logits.size(-1))
        _, topk_indices = torch.topk(logits, k=k_val, dim=-1, sorted=True)
        return logits, topk_indices
