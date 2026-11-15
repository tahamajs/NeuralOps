#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Scaled Dot-Product Attention Core Primitive.

Pure tensor-level and modular scaled dot-product attention implementing the canonical
attention equation Softmax(Q @ K^T / sqrt(d) + mask) @ V with robust numerical safeguards.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.layers.full_attention import FullAttention

__all__ = [
    "ScaledDotProductAttention",
    "scaled_dot_product_attention",
]


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Computes scaled dot-product attention with manual and fast-path options.

    Args:
        query: Query tensor (..., seq_len_q, head_dim).
        key: Key tensor (..., seq_len_k, head_dim).
        value: Value tensor (..., seq_len_k, head_dim).
        attn_mask: Optional attention mask broadcastable to (..., seq_len_q, seq_len_k).
        dropout_p: Dropout probability applied to attention weights.
        is_causal: Whether strict lower-triangular causal masking is enforced.
        scale: Scaling factor; defaults to 1.0 / sqrt(head_dim).

    Returns:
        Tuple of (context, attention_weights).
    """
    head_dim = query.shape[-1]
    inv_scale = scale if scale is not None else 1.0 / math.sqrt(max(head_dim, 1))

    # Fast path via PyTorch SDPA when weights are not explicitly required
    if attn_mask is None or attn_mask.dtype == torch.bool or attn_mask.dim() == 4:
        try:
            mask_arg = attn_mask.to(query.dtype) if (attn_mask is not None and attn_mask.dim() == 4) else None
            out = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=mask_arg,
                dropout_p=dropout_p,
                is_causal=is_causal and mask_arg is None,
                scale=inv_scale,
            )
            return out, None
        except Exception:
            pass

    # Mathematical reference path
    scores = torch.matmul(query.float(), key.float().transpose(-2, -1)) * inv_scale
    seq_q = query.shape[-2]
    seq_k = key.shape[-2]

    if is_causal:
        q_idx = torch.arange(seq_q, device=query.device).unsqueeze(1) + (seq_k - seq_q)
        k_idx = torch.arange(seq_k, device=query.device).unsqueeze(0)
        causal_mask = (k_idx > q_idx).unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(causal_mask, -1e4)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores = scores.masked_fill(~attn_mask, -1e4)
        else:
            scores = scores + attn_mask.to(scores.dtype)

    weights = F.softmax(scores, dim=-1).to(query.dtype)
    weights = torch.nan_to_num(weights, nan=0.0)
    if dropout_p > 0.0:
        weights = F.dropout(weights, p=dropout_p, training=True)

    out = torch.matmul(weights, value)
    return out, weights


class ScaledDotProductAttention(FullAttention):
    """Modular Scaled Dot-Product Attention operator."""

    FAMILY: str = "dense"
