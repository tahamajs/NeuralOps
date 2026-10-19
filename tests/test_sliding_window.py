#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Unit tests for sliding window attention."""
import torch
import pytest
from neuralops.attentions.modern.dynamic_conv_attention import SlidingWindowAttention


def test_sliding_window_forward_shape():
    layer = SlidingWindowAttention(window_size=128)
    x = torch.randn(2, 64, 32)
    out, attn = layer(x)
    assert out.shape == (2, 64, 32)
    assert attn is None


def test_sliding_window_clamping():
    layer = SlidingWindowAttention(window_size=32)
    x = torch.randn(2, 100, 32)
    out, attn = layer(x)
    assert out.shape == (2, 32, 32)
    assert torch.equal(out, x[:, -32:, :])


def test_sliding_window_with_mask():
    layer = SlidingWindowAttention(window_size=64)
    x = torch.randn(1, 32, 16)
    mask = torch.ones(1, 1, 32, 32)
    out, _ = layer(x, attention_mask=mask)
    assert out.shape == (1, 32, 16)
