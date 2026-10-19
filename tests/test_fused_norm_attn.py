# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for FusedNormAttention."""
import torch
import torch.nn as nn
from neuralops.core.fused_norm_attention import FusedNormAttention


class DummyAttn(nn.Module):
    def forward(self, x):
        return x * 0.5


def test_fused_norm_attention_rmsnorm():
    attn = DummyAttn()
    block = FusedNormAttention(attn, normalized_shape=16, norm_type="rmsnorm")
    x = torch.randn(2, 4, 16)
    out = block(x)
    assert out.shape == x.shape


def test_fused_norm_attention_layernorm():
    attn = DummyAttn()
    block = FusedNormAttention(attn, normalized_shape=16, norm_type="layernorm")
    x = torch.randn(2, 4, 16)
    out = block(x)
    assert out.shape == x.shape
