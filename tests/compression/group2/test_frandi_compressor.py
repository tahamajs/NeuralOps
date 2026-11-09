#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit tests for FRanDICompressor (Group 2, Key 34).
"""

import pytest
import torch
import torch.nn as nn
from neuralops.compression.low_rank.frandi_compressor import (
    FRanDICompressor,
    FRanDILinear,
    frandi_compressor_factory,
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
    assert "frandi_compressor" in COMPRESSOR_REGISTRY
    assert COMPRESSOR_REGISTRY.get("frandi_compressor") is not None


def test_frandi_linear():
    lin = nn.Linear(128, 64)
    layer = FRanDILinear.from_linear(lin, rank=16)
    assert layer.rank == 16
    assert layer.in_features == 128
    assert layer.out_features == 64

    x = torch.randn(2, 128)
    out = layer(x)
    assert out.shape == (2, 64)
    assert torch.isfinite(out).all()


def test_frandi_compressor_reduction():
    model = DummyMLP()
    compressor = FRanDICompressor(model, rank=16, max_error=1.0)
    compressed, stats = compressor.compress()
    assert compressed is not None
    assert stats["effective_parameter_ratio"] > 1.05

    x = torch.randn(2, 128)
    out = compressed(x)
    assert out.shape == (2, 64)
    assert torch.isfinite(out).all()


def test_frandi_factory():
    model = DummyMLP()
    compressed, stats = frandi_compressor_factory(model, rank=16, return_stats=True)
    assert compressed is not None
    assert "effective_parameter_ratio" in stats
