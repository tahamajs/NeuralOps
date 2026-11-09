#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit tests for DynamicEntropySVDCompressor (Group 2, Key 26).
"""

import pytest
import torch
import torch.nn as nn
from neuralops.compression.svd.dynamic_rank import (
    DynamicEntropySVDCompressor,
    DynamicSVDLinear,
    dynamic_rank_factory,
)
from neuralops.core.registry import COMPRESSOR_REGISTRY


class DummyMLP(nn.Module):
    def __init__(self, in_features: int = 128, hidden: int = 128, out_features: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(hidden, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def test_registry():
    assert "dynamic_rank" in COMPRESSOR_REGISTRY
    assert COMPRESSOR_REGISTRY.get("dynamic_rank") is not None


def test_dynamic_svd_linear():
    lin = nn.Linear(128, 64)
    layer = DynamicSVDLinear.from_linear(lin, rank=16)
    assert layer.rank == 16
    assert layer.in_features == 128
    assert layer.out_features == 64

    x = torch.randn(2, 128)
    out = layer(x)
    assert out.shape == (2, 64)
    assert torch.isfinite(out).all()


def test_dynamic_rank_compressor_reduction():
    model = DummyMLP()
    orig_params = sum(p.numel() for p in model.parameters())
    compressor = DynamicEntropySVDCompressor(model, min_rank=8, max_rank=32)
    compressed, stats = compressor.compress()
    assert compressed is not None
    comp_params = sum(p.numel() for p in compressed.parameters())
    assert (orig_params / comp_params) > 1.05

    x = torch.randn(2, 128)
    out = compressed(x)
    assert out.shape == (2, 64)
    assert torch.isfinite(out).all()


def test_dynamic_rank_factory():
    model = DummyMLP()
    compressed = dynamic_rank_factory(model, min_rank=8, max_rank=32)
    assert compressed is not None
