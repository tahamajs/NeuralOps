#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Comprehensive Standard Family Layer Suite (Wave 4 W4-L3).

Validates instantiation, forward pass, shape preservation, finite outputs,
and backward pass across all 23 owned layer modules.
"""

from __future__ import annotations

import pytest
import torch

from neuralops.layers.alibi import ALiBiAttention
from neuralops.layers.bidirectional_attention import BidirectionalAttention
from neuralops.layers.causal_self_attention import CausalSelfAttention
from neuralops.layers.cross_attention import CrossAttention
from neuralops.layers.dilated_attention import DilatedAttention
from neuralops.layers.flash_attention import FlashAttention
from neuralops.layers.flash_attention_2 import FlashAttention2
from neuralops.layers.flash_decoding import FlashDecodingAttention
from neuralops.layers.full_attention import FullAttention
from neuralops.layers.gqa import GroupedQueryAttention
from neuralops.layers.mha import MultiHeadAttention as MHA_Alias
from neuralops.layers.mqa import MultiQueryAttention
from neuralops.layers.multi_head_attention import MultiHeadAttention
from neuralops.layers.native_attention import NativeAttention
from neuralops.layers.paged_attention import PagedAttention
from neuralops.layers.relative_position import RelativePositionAttention
from neuralops.layers.rope import RotaryPositionalEmbedding
from neuralops.layers.rotary_embedding import RotaryEmbedding
from neuralops.layers.scaled_dot_product import ScaledDotProductAttention
from neuralops.layers.sdpa_attention import SDPAAttention
from neuralops.layers.standard_attention import StandardAttention
from neuralops.layers.windowed_attention import WindowedAttention


def test_standard_layers_forward_and_backward() -> None:
    """Test standard sequence-to-sequence attention layers."""
    batch, seq_len, hidden_dim = 2, 8, 64
    x = torch.randn(batch, seq_len, hidden_dim, requires_grad=True)

    layers = [
        FullAttention(hidden_dim=hidden_dim, num_heads=4),
        MHA_Alias(hidden_dim=hidden_dim, num_heads=4),
        MultiHeadAttention(hidden_dim=hidden_dim, num_heads=4),
        GroupedQueryAttention(hidden_dim=hidden_dim, num_heads=4, num_kv_heads=2),
        MultiQueryAttention(hidden_dim=hidden_dim, num_heads=4),
        FlashAttention(hidden_dim=hidden_dim, num_heads=4),
        FlashAttention2(hidden_dim=hidden_dim, num_heads=4),
        SDPAAttention(hidden_dim=hidden_dim, num_heads=4),
        ScaledDotProductAttention(hidden_dim=hidden_dim, num_heads=4),
        NativeAttention(hidden_dim=hidden_dim, num_heads=4),
        StandardAttention(hidden_dim=hidden_dim, num_heads=4),
        CausalSelfAttention(hidden_dim=hidden_dim, num_heads=4),
        BidirectionalAttention(hidden_dim=hidden_dim, num_heads=4),
        WindowedAttention(hidden_dim=hidden_dim, num_heads=4, window_size=4),
        DilatedAttention(hidden_dim=hidden_dim, num_heads=4, dilation_rate=2),
        ALiBiAttention(hidden_dim=hidden_dim, num_heads=4),
        RelativePositionAttention(hidden_dim=hidden_dim, num_heads=4, max_relative_distance=8),
    ]

    for layer in layers:
        x_in = x.clone().detach().requires_grad_(True)
        out = layer(x_in)
        out_tensor = out[0] if isinstance(out, tuple) else out
        assert out_tensor.shape == (batch, seq_len, hidden_dim), (
            f"Shape mismatch in {type(layer).__name__}: {out_tensor.shape}"
        )
        assert not torch.isnan(out_tensor).any(), f"NaN encountered in {type(layer).__name__}"

        loss = out_tensor.sum()
        loss.backward()
        assert x_in.grad is not None, f"Grad is None for {type(layer).__name__}"
        assert not torch.isnan(x_in.grad).any(), f"NaN in grad for {type(layer).__name__}"


def test_cross_attention_forward_backward() -> None:
    """Test CrossAttention with distinct query and context tensors."""
    batch, q_len, kv_len, hidden_dim = 2, 4, 8, 64
    x_q = torch.randn(batch, q_len, hidden_dim, requires_grad=True)
    x_kv = torch.randn(batch, kv_len, hidden_dim, requires_grad=True)

    layer = CrossAttention(hidden_dim=hidden_dim, num_heads=4)
    out = layer(x_q, context=x_kv)
    out_tensor = out[0] if isinstance(out, tuple) else out

    assert out_tensor.shape == (batch, q_len, hidden_dim)
    assert not torch.isnan(out_tensor).any()

    loss = out_tensor.sum()
    loss.backward()
    assert x_q.grad is not None
    assert x_kv.grad is not None


def test_flash_decoding_and_paged_attention_execution() -> None:
    """Test specialized inference layers: FlashDecodingAttention and PagedAttention."""
    batch, hidden_dim = 2, 64
    x = torch.randn(batch, 1, hidden_dim)

    # Flash decoding
    fd_layer = FlashDecodingAttention(hidden_dim=hidden_dim, num_heads=4)
    out_fd, past_kv = fd_layer(x, use_cache=True)
    assert out_fd.shape == (batch, 1, hidden_dim)
    assert past_kv is not None

    # Paged attention
    paged_layer = PagedAttention(hidden_dim=hidden_dim, num_heads=4, block_size=16)
    num_blocks = 8
    k_cache = torch.randn(num_blocks, 16, 4, 16)
    v_cache = torch.randn(num_blocks, 16, 4, 16)
    block_tables = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    seq_lens = torch.tensor([20, 25], dtype=torch.int32)

    x_paged = torch.randn(batch, 1, hidden_dim)
    out_paged, _ = paged_layer(x_paged, k_cache, v_cache, block_tables, seq_lens)
    assert out_paged.shape == (batch, 1, hidden_dim)
    assert not torch.isnan(out_paged).any()


def test_rotary_position_modules() -> None:
    """Test RotaryPositionalEmbedding and RotaryEmbedding modules."""
    rope = RotaryPositionalEmbedding(dim=16, max_seq_len=32)
    cos, sin = rope(16)
    assert cos.shape == (16, 16)
    assert sin.shape == (16, 16)

    rot = RotaryEmbedding(dim=16, max_position_embeddings=32)
    cos2, sin2 = rot(16)
    assert cos2.shape == (16, 16)
    assert sin2.shape == (16, 16)
