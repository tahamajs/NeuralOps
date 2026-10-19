# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for MultiQueryAttention."""
import torch
import pytest
from neuralops.attentions.modern.mqa import MultiQueryAttention
from neuralops.core.registry import ATTENTION_REGISTRY


def test_mqa_registry():
    assert "mqa_attention" in ATTENTION_REGISTRY


def test_mqa_forward():
    mod = MultiQueryAttention(embed_dim=64, num_heads=4)
    x = torch.randn(2, 8, 64)
    out, weights = mod(x)
    assert out.shape == (2, 8, 64)
    assert weights is None


def test_mqa_causal():
    mod = MultiQueryAttention(embed_dim=64, num_heads=4, causal=True)
    x = torch.randn(2, 12, 64)
    out, _ = mod(x)
    assert out.shape == (2, 12, 64)
