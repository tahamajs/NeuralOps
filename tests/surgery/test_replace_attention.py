#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for replace_attention surgery operator.
"""

import torch
import torch.nn as nn

from neuralops.surgery.attention.replace import replace_attention
from neuralops.surgery.registry import get_surgery


class MockDecoderLayer(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.Linear(dim * 4, dim))

    def forward(self, x):
        attn_out, _ = self.self_attn(x, x, x)
        return attn_out + self.mlp(x)


class MockLlamaModel(nn.Module):
    def __init__(self, num_layers=2, dim=64):
        super().__init__()
        self.layers = nn.ModuleList([MockDecoderLayer(dim) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def test_replace_attention_registered():
    meta = get_surgery("replace_attention")
    assert meta.name == "replace_attention"
    assert "llama" in meta.compat
    assert "qwen" in meta.compat


def test_replace_attention_execution():
    model = MockLlamaModel(num_layers=2, dim=32)
    x = torch.randn(2, 8, 32)
    orig_out = model(x)
    assert orig_out.shape == (2, 8, 32)

    # Execute replace_attention
    replaced = replace_attention(model, attention="chunk_gla")
    new_out = replaced(x)
    assert new_out.shape == (2, 8, 32)
