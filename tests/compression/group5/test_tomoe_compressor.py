#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit tests for ToMoECompressor (Group 5, Key 78).
"""

import pytest
import torch
import torch.nn as nn
from neuralops.compression.low_rank.tomoe_compressor import (
    LowRankExpert,
    ToMoECompressor,
    ToMoELinear,
    tomoe_factory,
)
from neuralops.core.registry import COMPRESSOR_REGISTRY


class DummyMLP(nn.Module):
    def __init__(self, in_features: int = 128, hidden: int = 256, out_features: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


def test_registry():
    assert "tomoe_compressor" in COMPRESSOR_REGISTRY
    assert COMPRESSOR_REGISTRY.get("tomoe_compressor") is not None


def test_tomoe_linear():
    lin = nn.Linear(128, 64)
    moe_layer = ToMoELinear(original=lin, num_experts=4, top_k=2, rank=8)
    assert moe_layer.num_experts == 4
    assert moe_layer.top_k == 2

    x = torch.randn(4, 128)
    out = moe_layer(x)
    assert out.shape == (4, 64)
    assert torch.isfinite(out).all()


def test_tomoe_compressor_reduction():
    model = DummyMLP()
    orig_params = sum(p.numel() for p in model.parameters())
    compressor = ToMoECompressor(model, num_experts=4, top_k=2, rank=8, max_error=1.0)
    compressed, stats = compressor.compress()

    comp_params = sum(p.numel() for p in compressed.parameters())
    assert orig_params / max(comp_params, 1) > 1.05

    x = torch.randn(2, 128)
    out = compressed(x)
    assert out.shape == (2, 64)
    assert torch.isfinite(out).all()


def test_tomoe_factory():
    model = DummyMLP()
    comp_model, stats = tomoe_factory(model, num_experts=4, top_k=2, rank=8, return_stats=True)
    assert comp_model is not None
    assert isinstance(stats, dict)
