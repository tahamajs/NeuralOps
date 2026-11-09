#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit tests for LIFTRCompressor & _LinearizedGELU (Group 3, Key 48).
"""

import sys
if sys.path and (sys.path[0].endswith("/tests") or sys.path[0] == "tests"):
    sys.path.pop(0)

import pytest
import torch
import torch.nn as nn
from neuralops.compression.experimental.liftr import LIFTRCompressor, _LinearizedGELU
from neuralops.core.registry import COMPRESSOR_REGISTRY


class DummyMLP(nn.Module):
    def __init__(self, in_features: int = 128, out_features: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(out_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


def test_liftr_registry():
    assert "liftr" in COMPRESSOR_REGISTRY
    assert COMPRESSOR_REGISTRY.get("liftr") is not None


def test_linearized_gelu():
    layer = _LinearizedGELU(alpha=0.8)
    x = torch.randn(2, 64)
    out = layer(x)
    assert torch.allclose(out, 0.8 * x)


def test_liftr_compress_and_forward():
    model = DummyMLP()
    orig_params = sum(p.numel() for p in model.parameters())

    compressor = LIFTRCompressor(model, gelu_coefficient=0.8)
    compressed, stats = compressor.compress()

    assert compressed is not None
    comp_params = sum(p.numel() for p in compressed.parameters())
    assert orig_params / comp_params > 1.05

    x = torch.randn(2, 128)
    out = compressed(x)
    assert out.shape == (2, 128)
    assert torch.isfinite(out).all()
