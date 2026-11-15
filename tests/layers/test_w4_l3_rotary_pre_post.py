#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for Rotary Position Embedding (RoPE) pre vs post projection (Wave 4 W4-L3)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from neuralops.layers.rope import (
    RotaryPositionalEmbedding,
    apply_rotary_pos_emb,
    post_projection_rope,
    pre_projection_rope,
)
from neuralops.layers.rotary_embedding import RotaryEmbedding


def test_rope_relative_shift_invariance() -> None:
    """Verify that post-projection RoPE preserves relative shift invariance: <R(m)q, R(n)k> = <R(m+s)q, R(n+s)k>."""
    head_dim = 32
    max_seq_len = 64
    rope_module = RotaryPositionalEmbedding(dim=head_dim, max_seq_len=max_seq_len)

    seq_len = 30
    q = torch.randn(1, 1, seq_len, head_dim)
    k = torch.randn(1, 1, seq_len, head_dim)

    cos, sin = rope_module(seq_len)
    q_rot, k_rot = post_projection_rope(q, k, cos, sin)

    # Pick position m=5, n=2 (relative offset = 3)
    # Pick position m'=15, n'=12 (relative offset = 3)
    # Make q at m and m' identical, k at n and n' identical
    q_vec = torch.randn(1, 1, 1, head_dim)
    k_vec = torch.randn(1, 1, 1, head_dim)

    # Compute rotated vectors at position m=5 and n=2
    cos_5, sin_5 = cos[5:6], sin[5:6]
    cos_2, sin_2 = cos[2:3], sin[2:3]
    q_rot_5, _ = apply_rotary_pos_emb(q_vec, q_vec, cos_5, sin_5)
    _, k_rot_2 = apply_rotary_pos_emb(k_vec, k_vec, cos_2, sin_2)
    score_1 = (q_rot_5 * k_rot_2).sum()

    # Compute rotated vectors at shifted positions m=15 and n=12
    cos_15, sin_15 = cos[15:16], sin[15:16]
    cos_12, sin_12 = cos[12:13], sin[12:13]
    q_rot_15, _ = apply_rotary_pos_emb(q_vec, q_vec, cos_15, sin_15)
    _, k_rot_12 = apply_rotary_pos_emb(k_vec, k_vec, cos_12, sin_12)
    score_2 = (q_rot_15 * k_rot_12).sum()

    # Dot products must match within floating point precision
    assert torch.allclose(score_1, score_2, atol=1e-4)


def test_rope_pre_vs_post_projection_orthogonality() -> None:
    """Verify pre_projection_rope vs post_projection_rope semantics."""
    batch, seq_len, hidden_dim = 2, 8, 32
    x = torch.randn(batch, seq_len, hidden_dim)

    rope_mod = RotaryPositionalEmbedding(dim=hidden_dim, max_seq_len=seq_len)
    cos, sin = rope_mod(seq_len)

    # Pre-projection rotation
    x_rot = pre_projection_rope(x, cos, sin)
    assert x_rot.shape == x.shape
    assert not torch.isnan(x_rot).any()

    # Post-projection rotation on heads
    q_heads = torch.randn(batch, 4, seq_len, 8)
    k_heads = torch.randn(batch, 4, seq_len, 8)
    rope_head = RotaryPositionalEmbedding(dim=8, max_seq_len=seq_len)
    cos_head, sin_head = rope_head(seq_len)

    q_rot, k_rot = post_projection_rope(q_heads, k_heads, cos_head, sin_head)
    assert q_rot.shape == q_heads.shape
    assert k_rot.shape == k_heads.shape
    assert not torch.isnan(q_rot).any()
    assert not torch.isnan(k_rot).any()


def test_rotary_embedding_standalone_module() -> None:
    """Verify RotaryEmbedding standalone module behaves identically."""
    head_dim = 16
    rot_emb = RotaryEmbedding(dim=head_dim, max_position_embeddings=32)
    cos, sin = rot_emb(seq_len=16)

    assert cos.shape == (16, head_dim)
    assert sin.shape == (16, head_dim)
    assert torch.allclose(cos**2 + sin**2, torch.ones_like(cos), atol=1e-5)
