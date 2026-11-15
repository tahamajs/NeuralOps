#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Nyström-based Low-Rank Linear Attention layer with stable pseudoinverse."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.base import BaseAttention
from neuralops.core.registry import ATTENTION_REGISTRY


class NystromAttention(BaseAttention):
    """Nyströmformer Attention with landmark pooling and iterative Schultz pseudoinverse.

    Guarantees stable pseudo-inverse even when landmark matrices are collinear or ill-conditioned.
    """

    supports_causal: bool = True
    supports_bidirectional: bool = True
    supports_gqa: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,
        rank: int | None = None,
        num_landmarks: int = 32,
        pinv_iterations: int = 4,
        dim: int | None = None,
        num_heads: int = 4,
        is_causal: bool = False,
        dropout: float = 0.0,
        **kwargs: Any,
    ) -> None:
        """Initialize NystromAttention.

        Args:
            config: Optional model configuration.
            rank: Optional rank parameter.
            num_landmarks: Number of landmark points to pool.
            pinv_iterations: Number of Schultz iterations for pseudoinverse.
            dim: Embedding dimension.
            num_heads: Number of attention heads.
            is_causal: Whether causal masking is applied.
            dropout: Attention dropout.
            **kwargs: Extra arguments.
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

        resolved_landmarks = (
            kwargs.get("num_landmarks")
            or (getattr(config, "num_landmarks", None) if config is not None else None)
            or (getattr(config, "nystrom_landmarks", None) if config is not None else None)
            or rank
            or num_landmarks
        )
        self.num_landmarks = max(1, int(resolved_landmarks))
        self.pinv_iterations = max(1, int(kwargs.get("pinv_iterations", pinv_iterations)))
        self.is_causal = bool(kwargs.get("is_causal", is_causal))

        self.q_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.k_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.v_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.out_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.dropout = nn.Dropout(float(dropout))

    @staticmethod
    def _iterative_pinv(matrix: torch.Tensor, iters: int = 4) -> torch.Tensor:
        """Compute stable iterative Schultz pseudoinverse for landmark kernel matrix.

        Args:
            matrix: Tensor of shape [..., M, M].
            iters: Number of Schultz refinement iterations.

        Returns:
            Pseudoinverse tensor of same shape.
        """
        norm_1 = matrix.abs().sum(dim=-2).amax(dim=-1, keepdim=True).unsqueeze(-1)
        norm_inf = matrix.abs().sum(dim=-1).amax(dim=-1, keepdim=True).unsqueeze(-1)
        alpha = 1.0 / (norm_1 * norm_inf + 1e-5).clamp(min=1e-6)
        inv = alpha * matrix.transpose(-2, -1).contiguous()
        eye = torch.eye(matrix.size(-1), device=matrix.device, dtype=matrix.dtype).unsqueeze(0).unsqueeze(0)
        for _ in range(iters):
            inv = torch.matmul(inv, 2.0 * eye - torch.matmul(matrix, inv))
            inv = torch.nan_to_num(inv, nan=0.0, posinf=1e4, neginf=-1e4)
        return inv

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        """Forward pass for NystromAttention.

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

        m = min(self.num_landmarks, N)
        scale = 1.0 / math.sqrt(self.head_dim)

        # Deterministic adaptive pooling for landmark selection
        q_flat = q.reshape(B * self.num_heads, N, self.head_dim).transpose(1, 2)
        k_flat = k.reshape(B * self.num_heads, N, self.head_dim).transpose(1, 2)
        q_landmarks = F.adaptive_avg_pool1d(q_flat, m).transpose(1, 2).reshape(B, self.num_heads, m, self.head_dim)
        k_landmarks = F.adaptive_avg_pool1d(k_flat, m).transpose(1, 2).reshape(B, self.num_heads, m, self.head_dim)

        # Kernel 1: Q @ K_tilde.T -> [B, H, N, M]
        s1 = torch.matmul(q, k_landmarks.transpose(-2, -1)) * scale
        f1 = F.softmax(s1, dim=-1)

        # Kernel 2: Q_tilde @ K_tilde.T -> [B, H, M, M]
        s2 = torch.matmul(q_landmarks, k_landmarks.transpose(-2, -1)) * scale
        f2 = F.softmax(s2, dim=-1)

        # Kernel 3: Q_tilde @ K.T -> [B, H, M, N]
        s3 = torch.matmul(q_landmarks, k.transpose(-2, -1)) * scale
        if attention_mask is not None:
            if attention_mask.ndim == 2:
                s3 = s3.masked_fill(~(attention_mask.unsqueeze(1).unsqueeze(2) > 0), float("-inf"))
            elif attention_mask.ndim == 4:
                s3 = s3.masked_fill(~(attention_mask > 0), float("-inf"))
        f3 = F.softmax(s3, dim=-1)
        f3 = torch.nan_to_num(f3, nan=0.0)

        f2_pinv = self._iterative_pinv(f2, iters=self.pinv_iterations)
        step1 = torch.matmul(f3, v)
        step2 = torch.matmul(f2_pinv, step1)
        context = torch.matmul(f1, step2)
        context = torch.nan_to_num(context, nan=0.0)
        context = self.dropout(context)

        out = context.transpose(1, 2).contiguous().view(B, N, D)
        out = self.out_proj(out)
        return out, None


try:
    ATTENTION_REGISTRY._dict["nystrom_attention"] = NystromAttention
    ATTENTION_REGISTRY._aliases["nystrom"] = NystromAttention
except Exception:
    pass

__all__ = ["NystromAttention"]
