#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit tests for DRONECompressor (Group 2, Key 25).
"""

import pytest
import torch
import torch.nn as nn
from neuralops.compression.low_rank.drone_compressor import (
    DRONECompressor,
)
from neuralops.core.registry import COMPRESSOR_REGISTRY


class DummyMLP(nn.Module):
    def __init__(self, in_features: int = 128, hidden: int = 128, out_features: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


def test_registry():
    assert "drone_compressor" in COMPRESSOR_REGISTRY
    assert COMPRESSOR_REGISTRY.get("drone_compressor") is not None


def test_drone_compressor_reduction():
    model = DummyMLP()
    compressor = DRONECompressor(model, rank=16, max_error=1.0)
    compressed, stats = compressor.compress()
    assert compressed is not None
    assert stats["effective_parameter_ratio"] > 1.05

    x = torch.randn(2, 128)
    out = compressed(x)
    assert out.shape == (2, 64)
    assert torch.isfinite(out).all()


def test_drone_compressor_options():
    model = DummyMLP()
    compressor = DRONECompressor(model, rank=8, max_error=0.8)
    compressed, stats = compressor.compress()
    assert compressed is not None
    assert "effective_parameter_ratio" in stats
