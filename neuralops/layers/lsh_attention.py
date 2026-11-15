#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Locality-Sensitive Hashing (LSH) Attention with configuration-compliant bucketing."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.base import BaseAttention
from neuralops.core.registry import ATTENTION_REGISTRY


class LSHAttention(BaseAttention):
    """Reformer-style Locality-Sensitive Hashing (LSH) Attention.

    Assigns query and key vectors to discrete hash buckets using random hyperplane projections.
    """

    supports_causal: bool = True
    supports_bidirectional: bool = True
    supports_gqa: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,
        rank: int | None = None,
        rounds: int = 4,
        num_buckets: int = 16,
        dim: int | None = None,
        num_heads: int = 4,
        is_causal: bool = False,
        dropout: float = 0.0,
        **kwargs: Any,
    ) -> None:
        """Initialize LSHAttention.

        Args:
            config: Optional configuration object.
            rank: Optional rank parameter.
            rounds: Number of independent hashing rounds.
            num_buckets: Number of discrete hash buckets per round.
            dim: Total embedding dimension.
            num_heads: Number of attention heads.
            is_causal: Whether causal masking is enforced.
            dropout: Attention weight dropout.
            **kwargs: Extra arguments.
        """
        super().__init__(config=config if not isinstance(config, int) else None, **kwargs)

        resolved_buckets = (
            kwargs.get("lsh_buckets")
            or kwargs.get("lsh_num_buckets")
            or kwargs.get("num_buckets")
            or (getattr(config, "lsh_buckets", None) if config is not None else None)
            or (getattr(config, "lsh_num_buckets", None) if config is not None else None)
            or (getattr(config, "num_buckets", None) if config is not None else None)
            or num_buckets
        )
        self.num_buckets = max(2, int(resolved_buckets))

        resolved_rounds = (
            kwargs.get("rounds")
            or (getattr(config, "lsh_rounds", None) if config is not None else None)
            or (getattr(config, "rounds", None) if config is not None else None)
            or rounds
        )
        self.rounds = max(1, int(resolved_rounds))

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
        self.is_causal = bool(kwargs.get("is_causal", is_causal))

        self.q_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.k_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.v_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.out_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.dropout = nn.Dropout(float(dropout))

        half_buckets = max(1, self.num_buckets // 2)
        proj_data = torch.randn(self.rounds, self.head_dim, half_buckets)
        proj_data = F.normalize(proj_data, p=2, dim=1)
        self.register_buffer("proj_matrices", proj_data, persistent=False)

    def _hash(self, vectors: torch.Tensor, proj: torch.Tensor) -> torch.Tensor:
        """Hash vectors into bucket indices [0, num_buckets).

        Args:
            vectors: Tensor of shape [B, H, N, head_dim].
            proj: Projection matrix [head_dim, half_buckets].

        Returns:
            Bucket indices tensor of shape [B, H, N].
        """
        norm_v = F.normalize(vectors.float(), p=2, dim=-1)
        dots = torch.matmul(norm_v, proj.float())
        signs = (dots >= 0).long()
        num_planes = signs.shape[-1]
        weights = (2 ** torch.arange(num_planes, device=signs.device)).long()
        bucket = (signs * weights).sum(dim=-1)
        return bucket % self.num_buckets

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        """Forward pass for LSHAttention.

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

        scale = 1.0 / math.sqrt(self.head_dim)
        accum = torch.zeros_like(q)

        q_idx = torch.arange(N, device=x.device).unsqueeze(1)
        k_idx = torch.arange(N, device=x.device).unsqueeze(0)
        causal_mask = (q_idx >= k_idx).unsqueeze(0).unsqueeze(0) if self.is_causal else None

        for r in range(self.rounds):
            proj_r = self.proj_matrices[r].to(device=x.device, dtype=q.dtype)
            q_hash = self._hash(q, proj_r)
            k_hash = self._hash(k, proj_r)

            # Match tokens in the same hash bucket
            match = q_hash.unsqueeze(-1) == k_hash.unsqueeze(-2)  # [B, H, N, N]
            if causal_mask is not None:
                match = match & causal_mask
            if attention_mask is not None:
                if attention_mask.ndim == 2:
                    match = match & (attention_mask.unsqueeze(1).unsqueeze(2) > 0)
                elif attention_mask.ndim == 4:
                    match = match & (attention_mask > 0)

            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            scores = scores.masked_fill(~match, float("-inf"))
            weights = F.softmax(scores, dim=-1)
            weights = torch.nan_to_num(weights, nan=0.0)
            weights = self.dropout(weights)
            accum = accum + torch.matmul(weights, v)

        accum = accum / float(self.rounds)
        out = accum.transpose(1, 2).contiguous().view(B, N, D)
        out = self.out_proj(out)
        return out, None


try:
    ATTENTION_REGISTRY._dict["lsh_attention"] = LSHAttention
    ATTENTION_REGISTRY._aliases["lsh"] = LSHAttention
except Exception:
    pass

__all__ = ["LSHAttention"]
