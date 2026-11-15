#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Regression tests for GQA and MQA KV head expansion semantics."""

import pytest
import torch

from neuralops.layers.gqa import GroupedQueryAttention, expand_kv_heads
from neuralops.layers.mqa import MultiQueryAttention


def test_gqa_repeat_interleave_vs_repeat_semantics():
    """Verify num_kv_heads expansion uses repeat_interleave, not repeat.

    With num_heads=8 and num_kv_heads=2, repetition factor is 4.
    Input KV heads: [head0, head1].
    Correct GQA layout: [head0, head0, head0, head0, head1, head1, head1, head1].
    Incorrect .repeat() layout: [head0, head1, head0, head1, head0, head1, head0, head1].
    """
    b, num_kv_heads, seq_len, head_dim = 2, 2, 4, 16
    kv = torch.empty(b, num_kv_heads, seq_len, head_dim)
    kv[:, 0, :, :].fill_(10.0)  # Head 0 has values 10.0
    kv[:, 1, :, :].fill_(20.0)  # Head 1 has values 20.0

    n_rep = 4  # num_heads = 8
    expanded = expand_kv_heads(kv, n_rep)

    assert expanded.shape == (b, 8, seq_len, head_dim)

    # First 4 heads must belong to KV head 0 (values == 10.0)
    for h in range(4):
        assert torch.all(expanded[:, h] == 10.0), f"Head {h} should belong to KV head 0"

    # Next 4 heads must belong to KV head 1 (values == 20.0)
    for h in range(4, 8):
        assert torch.all(expanded[:, h] == 20.0), f"Head {h} should belong to KV head 1"

    # Confirm .repeat() does NOT match correct GQA grouping
    wrong_repeat = kv.repeat(1, n_rep, 1, 1)
    assert not torch.equal(expanded, wrong_repeat), "repeat() must not equal repeat_interleave()"


def test_gqa_forward_num_heads_8_num_kv_heads_2():
    """Test GQA layer forward with num_heads=8, num_kv_heads=2."""
    b, seq_len, dim = 2, 16, 64
    layer = GroupedQueryAttention(dim=dim, num_heads=8, num_kv_heads=2)
    assert layer.num_heads == 8
    assert layer.num_kv_heads == 2
    assert layer.kv_repeat == 4

    x = torch.randn(b, seq_len, dim, requires_grad=True)
    out, _ = layer(x)

    assert out.shape == (b, seq_len, dim)
    assert torch.isfinite(out).all()

    # Verify backward propagation
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_mqa_single_kv_head_expansion():
    """Test MQA forward pass with num_heads=8 and strictly 1 KV head."""
    b, seq_len, dim = 2, 16, 64
    layer = MultiQueryAttention(dim=dim, num_heads=8)
    assert layer.num_heads == 8
    assert layer.num_kv_heads == 1

    x = torch.randn(b, seq_len, dim, requires_grad=True)
    out, _ = layer(x)

    assert out.shape == (b, seq_len, dim)
    assert torch.isfinite(out).all()

    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
