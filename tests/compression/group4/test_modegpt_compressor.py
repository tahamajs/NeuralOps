#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit tests for MoDeGPTCompressor (Group 4, Key 54: modegpt_compressor).
"""

import pytest
import torch
import torch.nn as nn
from neuralops.compression.low_rank.modegpt_compressor import (
    MoDeGPTCompressor,
    modegpt_factory,
    MoDeGPTLinear,
)
from neuralops.core.registry import COMPRESSOR_REGISTRY


class DummyMLP(nn.Module):
    def __init__(self, in_features: int = 128, out_features: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(out_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


def test_registry():
    assert "modegpt_compressor" in COMPRESSOR_REGISTRY
    assert COMPRESSOR_REGISTRY.get("modegpt_compressor") is not None


def test_modegpt_linear_forward():
    lin = nn.Linear(128, 128)
    layer = MoDeGPTLinear.from_linear(lin, rank=16)
    x = torch.randn(2, 128)
    out = layer(x)
    assert out.shape == (2, 128)
    assert torch.isfinite(out).all()


def test_modegpt_compressor():
    model = DummyMLP()
    comp = MoDeGPTCompressor(model, rank=16)
    compressed, stats = comp.compress()
    assert compressed is not None
    assert stats["forward_check_success"] is True

    x = torch.randn(2, 128)
    out = compressed(x)
    assert out.shape == (2, 128)
    assert torch.isfinite(out).all()


def test_modegpt_factory():
    model = DummyMLP()
    compressed = modegpt_factory(model, rank=16)
    assert compressed is not None
    x = torch.randn(2, 128)
    out = compressed(x)
    assert torch.isfinite(out).all()
