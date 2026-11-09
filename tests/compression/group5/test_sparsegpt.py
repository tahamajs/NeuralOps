#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit tests for SparseGPT (Group 5, Key 72).
"""

import pytest
import torch
import torch.nn as nn
from neuralops.compression.pruning.sparsegpt import (
    SparseGPTCompressor,
    sparsegpt_factory,
    sparsegpt_prune_layer,
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
    assert "sparsegpt" in COMPRESSOR_REGISTRY
    assert COMPRESSOR_REGISTRY.get("sparsegpt") is not None


def test_sparsegpt_prune_layer():
    torch.manual_seed(42)
    lin = nn.Linear(64, 64)
    mod, mask = sparsegpt_prune_layer(lin, sparsity=0.5)

    assert mask.shape == (64, 64)
    zeros_fraction = float((mod.weight.data == 0).sum()) / mod.weight.data.numel()
    assert 0.45 <= zeros_fraction <= 0.55

    x = torch.randn(4, 64)
    out = mod(x)
    assert out.shape == (4, 64)
    assert torch.isfinite(out).all()


def test_sparsegpt_compressor():
    model = DummyMLP()
    compressor = SparseGPTCompressor(model, sparsity=0.5)
    compressed, stats = compressor.compress()

    assert stats.get("sparsity", 0.0) >= 0.40
    x = torch.randn(2, 128)
    out = compressed(x)
    assert out.shape == (2, 64)
    assert torch.isfinite(out).all()


def test_sparsegpt_factory():
    model = DummyMLP()
    comp_model, stats = sparsegpt_factory(model, sparsity=0.5, return_stats=True)
    assert comp_model is not None
    assert isinstance(stats, dict)
