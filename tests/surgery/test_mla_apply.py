#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for apply_mla_attention_surgery.
Ensures AttentionTargetSpec attributes (attr_name and embed_dim) are used properly.
"""

import torch.nn as nn

from neuralops.surgery.mla_surgery import DeepSeekMLASurgeryBridge
from neuralops.surgery.pipeline import apply_mla_attention_surgery


class MockDecoderLayer(nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.Linear(dim * 4, dim))


class MockModel(nn.Module):
    def __init__(self, num_layers: int = 2, dim: int = 64):
        super().__init__()
        self.layers = nn.ModuleList([MockDecoderLayer(dim) for _ in range(num_layers)])


def test_apply_mla_attention_surgery_replaces_layers():
    """Verify apply_mla_attention_surgery uses attr_name and embed_dim to replace attention blocks."""
    model = MockModel(num_layers=2, dim=64)
    surgeried = apply_mla_attention_surgery(model, latent_kv_dim=32, decoupled_rope_dim=16)

    assert isinstance(surgeried.layers[0].self_attn, DeepSeekMLASurgeryBridge)
    assert isinstance(surgeried.layers[1].self_attn, DeepSeekMLASurgeryBridge)
    assert surgeried.layers[0].self_attn.hidden_dim == 64
