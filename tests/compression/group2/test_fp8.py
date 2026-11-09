#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit tests for FP8Compressor (Group 2, Key 33).
"""

import pytest
import torch
import torch.nn as nn
from neuralops.compression.quantization.fp8_compressor import (
    FP8Compressor,
    fp8_compressor_factory,
)
from neuralops.core.registry import COMPRESSOR_REGISTRY
from neuralops.ops.cuda.fp8_compressor import FP8Function


class DummyMLP(nn.Module):
    def __init__(self, in_features: int = 128, hidden: int = 128, out_features: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(hidden, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def test_registry():
    assert "fp8" in COMPRESSOR_REGISTRY
    assert COMPRESSOR_REGISTRY.get("fp8") is not None


def test_fp8_function_fallback():
    W = torch.randn(64, 128)
    scale = float(W.abs().max() / 448.0)
    out = FP8Function.apply(W, scale)
    assert out.shape == W.shape
    assert torch.isfinite(out).all()


def test_fp8_compressor_reduction():
    model = DummyMLP()
    compressor = FP8Compressor(model, fp8_format="e4m3")
    compressed, stats = compressor.compress()
    assert compressed is not None
    assert stats["bit_reduction"] == 4.0

    x = torch.randn(2, 128)
    out = compressed(x)
    assert out.shape == (2, 64)
    assert torch.isfinite(out).all()


def test_fp8_factory():
    model = DummyMLP()
    compressed = fp8_compressor_factory(model, fp8_format="e4m3")
    assert compressed is not None
