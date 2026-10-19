#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Unit tests for ALiBi attention."""
import torch
import pytest
from neuralops.attentions.modern.alibi import ALiBiAttention, get_alibi_slopes


def test_alibi_slopes():
    slopes = get_alibi_slopes(8)
    assert slopes.shape == (8,)
    assert (slopes > 0).all()


def test_alibi_slopes_non_pow2():
    slopes = get_alibi_slopes(12)
    assert slopes.shape == (12,)
    assert (slopes > 0).all()


def test_alibi_forward():
    layer = ALiBiAttention(num_heads=8, hidden_size=64)
    x = torch.randn(2, 16, 64)
    out, attn = layer(x)
    assert out.shape == (2, 16, 64)
    assert attn is None
