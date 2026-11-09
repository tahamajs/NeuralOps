#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit tests for Wanda (Group 5, Key 81).
"""

import pytest
import torch
import torch.nn as nn
from neuralops.compression.pruning.wanda import (
    WandaCompressor,
    wanda_factory,
    wanda_prune_layer,
    wanda_prune_model,
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
    assert "wanda" in COMPRESSOR_REGISTRY
    assert COMPRESSOR_REGISTRY.get("wanda") is not None


def test_wanda_prune_layer():
    torch.manual_seed(42)
    lin = nn.Linear(64, 64)
    x = torch.randn(8, 64)
    x[:, 0] *= 50.0

    mod, mask = wanda_prune_layer(lin, sparsity=0.5, calib_data=x)
    assert mask.shape == (64, 64)
    zeros_fraction = float((mod.weight.data == 0).sum()) / mod.weight.data.numel()
    assert 0.45 <= zeros_fraction <= 0.55

    # Outlier column should be preserved
    assert mask[:, 0].sum() > 32

    out = mod(x)
    assert out.shape == (8, 64)
    assert torch.isfinite(out).all()


def test_wanda_compressor():
    model = DummyMLP()
    compressor = WandaCompressor(model, sparsity=0.5)
    compressed, stats = compressor.compress()

    assert stats.get("sparsity", 0.0) >= 0.40
    x = torch.randn(2, 128)
    out = compressed(x)
    assert out.shape == (2, 64)
    assert torch.isfinite(out).all()


def test_wanda_factory():
    model = DummyMLP()
    comp_model, stats = wanda_factory(model, sparsity=0.5, return_stats=True)
    assert comp_model is not None
    assert isinstance(stats, dict)
