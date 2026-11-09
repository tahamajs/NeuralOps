#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit tests for SmoothQuant (Group 5, Key 70).
"""

import pytest
import torch
import torch.nn as nn
from neuralops.compression.quantization.smoothquant import (
    SmoothQuantCompressor,
    SmoothQuantLinear,
    smoothquant_factory,
    smoothquant_layer,
    smoothquant_model,
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
    assert "smoothquant" in COMPRESSOR_REGISTRY
    assert COMPRESSOR_REGISTRY.get("smoothquant") is not None


def test_smoothquant_layer():
    torch.manual_seed(42)
    layer = nn.Linear(64, 64)
    x = torch.randn(8, 64)
    x[:, 5] *= 10.0  # simulate activation outlier

    mod, s = smoothquant_layer(layer, alpha=0.5, calib_data=x, quantize=True)
    assert hasattr(mod, "weight_int8")
    assert mod.weight_int8.dtype == torch.int8

    out = mod(x)
    assert out.shape == (8, 64)
    assert torch.isfinite(out).all()


def test_smoothquant_compressor():
    model = DummyMLP()
    compressor = SmoothQuantCompressor(model, alpha=0.5)
    compressed, stats = compressor.compress()

    assert stats.get("bit_reduction", 1.0) >= 4.0
    x = torch.randn(2, 128)
    out = compressed(x)
    assert out.shape == (2, 64)
    assert torch.isfinite(out).all()


def test_smoothquant_factory():
    model = DummyMLP()
    comp_model, stats = smoothquant_factory(model, alpha=0.5, return_stats=True)
    assert comp_model is not None
    assert isinstance(stats, dict)
