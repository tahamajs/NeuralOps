#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Block-Sparse Attention layer with deterministic block routing and stable top-k selection."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.base import BaseAttention
from neuralops.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register(
    "block_sparse",
    supports_causal=True,
    supports_bidirectional=True,
    supports_gqa=True,
)
@ATTENTION_REGISTRY.register(
    "block_sparse_attention",
    is_alias=True,
    supports_causal=True,
    supports_bidirectional=True,
    supports_gqa=True,
)
class BlockSparseAttention(BaseAttention):
    """Block-Sparse Attention module with deterministic block routing."""

    supports_causal: bool = True
    supports_bidirectional: bool = True
    supports_gqa: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,
        block_size: int = 16,
        top_k: int = 2,
        dim: int | None = None,
        num_heads: int = 4,
        is_causal: bool = False,
        dropout: float = 0.0,
        **kwargs: Any,
    ) -> None:
        """Initialize BlockSparseAttention.

        Args:
            config: Optional model configuration.
            block_size: Number of tokens per block.
            top_k: Number of key blocks routed per query block.
            dim: Embedding dimension.
            num_heads: Number of attention heads.
            is_causal: If True, enforce causal block lower-triangular routing.
            dropout: Attention dropout.
            **kwargs: Extra hyperparameters.
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

        self.block_size = max(1, int(kwargs.get("block_size", block_size)))
        self.top_k = max(1, int(kwargs.get("top_k", top_k)))
        self.is_causal = bool(kwargs.get("is_causal", is_causal))

        self.q_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.k_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.v_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.out_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.dropout = nn.Dropout(float(dropout))

    def _route_blocks(
        self,
        qb: torch.Tensor,
        kb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute block-level similarity and deterministically select top-k blocks.

        Args:
            qb: Blocked query representations [B, H, num_blocks, head_dim].
            kb: Blocked key representations [B, H, num_blocks, head_dim].

        Returns:
            Tuple of (block_scores, topk_indices) of shape [B, H, num_blocks, top_k].
        """
        B, H, nb, _ = qb.shape
        scale = 1.0 / math.sqrt(self.head_dim)
        block_scores = torch.matmul(qb, kb.transpose(-2, -1)) * scale  # [B, H, nb, nb]

        if self.is_causal:
            idx = torch.arange(nb, device=qb.device)
            causal_mask = idx.unsqueeze(1) >= idx.unsqueeze(0)  # [nb, nb]
            block_scores = block_scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        eff_k = min(self.top_k, nb)
        sorted_scores, topk_idx = torch.topk(block_scores, k=eff_k, dim=-1, sorted=True)
        return sorted_scores, topk_idx

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        """Forward pass for BlockSparseAttention.

        Args:
            x: Input tensor [B, N, D].
            attention_mask: Optional padding mask.
            **kwargs: Extra runtime options.

        Returns:
            Tuple of (output, None).
        """
        B, N, D = x.shape
        bs = self.block_size
        pad_len = (bs - (N % bs)) % bs
        if pad_len > 0:
            x_padded = F.pad(x, (0, 0, 0, pad_len))
        else:
            x_padded = x

        total_tokens = x_padded.size(1)
        nb = total_tokens // bs

        q = self.q_proj(x_padded).view(B, total_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_padded).view(B, total_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_padded).view(B, total_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        qb = q.view(B, self.num_heads, nb, bs, self.head_dim).mean(dim=3)
        kb = k.view(B, self.num_heads, nb, bs, self.head_dim).mean(dim=3)

        _, topk_idx = self._route_blocks(qb, kb)

        # Build token-level sparse block mask from top-k routing
        sparse_mask = torch.zeros(B, self.num_heads, nb, nb, dtype=torch.bool, device=x.device)
        sparse_mask.scatter_(dim=-1, index=topk_idx, value=True)

        token_mask = sparse_mask.repeat_interleave(bs, dim=2).repeat_interleave(bs, dim=3)
        token_mask = token_mask[:, :, :N, :N]

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q[:, :, :N], k[:, :, :N].transpose(-2, -1)) * scale

        if self.is_causal:
            c_idx = torch.arange(N, device=x.device)
            c_mask = c_idx.unsqueeze(1) >= c_idx.unsqueeze(0)
            token_mask = token_mask & c_mask.unsqueeze(0).unsqueeze(0)

        if attention_mask is not None:
            if attention_mask.ndim == 2:
                token_mask = token_mask & (attention_mask.unsqueeze(1).unsqueeze(2) > 0)
            elif attention_mask.ndim == 4:
                token_mask = token_mask & (attention_mask > 0)

        scores = scores.masked_fill(~token_mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        weights = self.dropout(weights)

        context = torch.matmul(weights, v[:, :, :N])
        out = context.transpose(1, 2).contiguous().view(B, N, D)
        out = self.out_proj(out)
        return out, None
