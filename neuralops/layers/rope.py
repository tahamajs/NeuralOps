#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Rotary Position Embedding (RoPE) Layer & Utilities.

Provides mathematical rotary position embeddings (Su et al., 2021) with explicit
separation and verification of post-projection (standard canonical per-head) versus
pre-projection rotations.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimensions of the input tensor."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Applies rotary position embeddings post-projection to query and key head tensors.

    Args:
        q: Projected query tensor of shape (batch, heads, seq_len, head_dim).
        k: Projected key tensor of shape (batch, heads_kv, seq_len, head_dim).
        cos: Precomputed cosine table broadcastable to (..., seq_len, head_dim).
        sin: Precomputed sine table broadcastable to (..., seq_len, head_dim).
        position_ids: Optional token position indices of shape (batch, seq_len).

    Returns:
        Tuple of rotated (q_rot, k_rot) matching input shapes and dtypes.
    """
    if position_ids is not None:
        if cos.dim() == 2:
            # (max_seq, dim) -> index by position_ids
            cos = cos[position_ids].unsqueeze(1)  # (b, 1, s, d)
            sin = sin[position_ids].unsqueeze(1)
        elif cos.dim() == 4 and cos.shape[0] == 1 and position_ids.shape[0] > 1:
            cos = cos.expand(position_ids.shape[0], -1, -1, -1)
            sin = sin.expand(position_ids.shape[0], -1, -1, -1)
    else:
        while cos.dim() < q.dim():
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)

    # Match sequence length dimension
    seq_len = q.shape[-2]
    cos_q = cos[..., :seq_len, : q.shape[-1]].to(q.dtype)
    sin_q = sin[..., :seq_len, : q.shape[-1]].to(q.dtype)

    seq_len_k = k.shape[-2]
    cos_k = cos[..., :seq_len_k, : k.shape[-1]].to(k.dtype)
    sin_k = sin[..., :seq_len_k, : k.shape[-1]].to(k.dtype)

    q_embed = (q * cos_q) + (_rotate_half(q) * sin_q)
    k_embed = (k * cos_k) + (_rotate_half(k) * sin_k)
    return q_embed, k_embed


def post_projection_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Canonical RoPE application: applied to query and key heads AFTER linear projection.

    Preserves rotational relative coordinate invariance because inner products
    <R(m) q, R(n) k> depend solely on the relative difference (m - n).
    """
    return apply_rotary_pos_emb(q, k, cos, sin, position_ids)


def pre_projection_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Pre-projection RoPE: rotates raw hidden states prior to linear projection.

    Warning:
        Pre-projection RoPE only preserves relative positional attention angles
        if the projection matrices W_q and W_k are block-orthogonal. Use
        `post_projection_rope` for standard Transformer architectures.
    """
    while cos.dim() < x.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

    seq_len = x.shape[-2]
    cos_x = cos[..., :seq_len, : x.shape[-1]].to(x.dtype)
    sin_x = sin[..., :seq_len, : x.shape[-1]].to(x.dtype)

    return (x * cos_x) + (_rotate_half(x) * sin_x)


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Positional Embedding (RoPE) generator module.

    Computes inverse frequency angles and generates cached (cos, sin) tables.
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 4096,
        base: float = 10000.0,
        device: torch.device | None = None,
        max_seq_len: int | None = None,
        **kwargs: Any,  # reason: absorbs unused kwargs
    ) -> None:
        """Initializes RotaryPositionalEmbedding."""
        super().__init__()
        self.dim: int = dim
        self.max_position_embeddings: int = int(max_seq_len or max_position_embeddings)
        self.base: float = float(base)

        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(self.max_position_embeddings, device=device, dtype=torch.float32)

    def _set_cos_sin_cache(self, seq_len: int, device: torch.device | None, dtype: torch.dtype) -> None:
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(
        self,
        x: torch.Tensor | int | None = None,
        seq_len: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieves cached (cos, sin) tables up to the specified sequence length."""
        if isinstance(x, int):
            target_seq = x
            dev = self.inv_freq.device
            dt = torch.float32
        elif isinstance(x, torch.Tensor):
            target_seq = seq_len if seq_len is not None else x.shape[-2]
            dev = x.device
            dt = x.dtype
        else:
            target_seq = seq_len if seq_len is not None else self.max_position_embeddings
            dev = self.inv_freq.device
            dt = torch.float32

        if target_seq > self.max_position_embeddings:
            self.max_position_embeddings = target_seq
            self._set_cos_sin_cache(target_seq, device=dev, dtype=dt)
        return (
            self.cos_cached[:target_seq].to(dtype=dt, device=dev),
            self.sin_cached[:target_seq].to(dtype=dt, device=dev),
        )


RoPE = RotaryPositionalEmbedding
apply_rope = apply_rotary_pos_emb

__all__ = [
    "RoPE",
    "RotaryPositionalEmbedding",
    "apply_rope",
    "apply_rotary_pos_emb",
    "post_projection_rope",
    "pre_projection_rope",
]
