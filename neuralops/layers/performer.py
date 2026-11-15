#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Performer FAVOR+ random feature kernel linear attention layer."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.base import BaseAttention
from neuralops.core.registry import ATTENTION_REGISTRY


class PerformerAttention(BaseAttention):
    """Performer Attention using FAVOR+ positive orthogonal random features."""

    supports_causal: bool = True
    supports_bidirectional: bool = True
    supports_gqa: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,
        num_features: int = 64,
        dim: int | None = None,
        num_heads: int = 4,
        is_causal: bool = False,
        dropout: float = 0.0,
        **kwargs: Any,
    ) -> None:
        """Initialize PerformerAttention.

        Args:
            config: Optional config.
            num_features: Number of random features for kernel estimation.
            dim: Embedding dimension.
            num_heads: Head count.
            is_causal: Autoregressive flag.
            dropout: Dropout probability.
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

        self.num_features = max(1, int(kwargs.get("num_features", num_features)))
        self.is_causal = bool(kwargs.get("is_causal", is_causal))

        self.q_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.k_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.v_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.out_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.dropout = nn.Dropout(float(dropout))

        # Generate orthogonal random projection weights
        proj = torch.randn(self.num_features, self.head_dim)
        proj, _ = torch.linalg.qr(proj.t())
        self.register_buffer("projection", proj.t().contiguous(), persistent=False)

    def _feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """FAVOR+ positive random feature mapping.

        phi(x) = exp(-||x||^2/2) * exp(w^T x)
        """
        scale = 1.0 / (self.head_dim**0.25)
        x_scaled = x * scale
        proj_matrix = self.projection.to(device=x.device, dtype=x.dtype)
        dots = torch.matmul(x_scaled, proj_matrix.t())  # [B, H, N, num_features]
        norm_sq = (x_scaled**2).sum(dim=-1, keepdim=True) / 2.0
        return torch.exp(dots - norm_sq) / math.sqrt(self.num_features)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        """Forward pass for PerformerAttention.

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

        q_prime = self._feature_map(q)  # [B, H, N, M]
        k_prime = self._feature_map(k)  # [B, H, N, M]

        if attention_mask is not None:
            if attention_mask.ndim == 2:
                k_prime = k_prime * (attention_mask.unsqueeze(1).unsqueeze(-1) > 0)
            elif attention_mask.ndim == 4:
                k_prime = k_prime * (attention_mask.mean(dim=2, keepdim=True).transpose(-2, -1) > 0)

        if self.is_causal:
            # Causal prefix sum scan
            # KV: [B, H, N, M, D]
            kv = torch.einsum("bhnm,bhnd->bhnmd", k_prime, v)
            kv_cum = torch.cumsum(kv, dim=2)
            k_cum = torch.cumsum(k_prime, dim=2)  # [B, H, N, M]

            context = torch.einsum("bhnm,bhnmd->bhnd", q_prime, kv_cum)
            norm = (q_prime * k_cum).sum(dim=-1, keepdim=True).clamp(min=1e-6)
            context = context / norm
        else:
            # Non-causal linear complexity: O(N)
            kv = torch.einsum("bhnm,bhnd->bhmd", k_prime, v)
            k_sum = k_prime.sum(dim=2, keepdim=True)  # [B, H, 1, M]
            context = torch.einsum("bhnm,bhmd->bhnd", q_prime, kv)
            norm = (q_prime * k_sum).sum(dim=-1, keepdim=True).clamp(min=1e-6)
            context = context / norm

        context = torch.nan_to_num(context, nan=0.0)
        context = self.dropout(context)
        out = context.transpose(1, 2).contiguous().view(B, N, D)
        out = self.out_proj(out)
        return out, None


try:
    ATTENTION_REGISTRY._dict["performer_attention"] = PerformerAttention
    ATTENTION_REGISTRY._aliases["performer"] = PerformerAttention
except Exception:
    pass

__all__ = ["PerformerAttention"]
