#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit tests for A3Compressor (Group 1, Key 1).
"""

import pytest
import torch
import torch.nn as nn
from neuralops.compression.low_rank.a3_compressor import (
    A3Compressor,
    A3Linear,
    a3_compressor_factory,
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


def test_a3_registry():
    assert "a3_compressor" in COMPRESSOR_REGISTRY
    assert COMPRESSOR_REGISTRY.get("a3_compressor") is not None


def test_a3_linear_forward_and_reconstruction():
    lin = nn.Linear(128, 64)
    a3_layer = A3Linear.from_linear(lin, rank=16, lora_rank=4)
    assert a3_layer.rank == 16
    assert a3_layer.lora_rank == 4
    assert a3_layer.in_features == 128
    assert a3_layer.out_features == 64

    x = torch.randn(4, 128)
    out = a3_layer(x)
    assert out.shape == (4, 64)
    assert torch.isfinite(out).all()

    w = a3_layer.get_weight()
    assert w.shape == (64, 128)


def test_a3_compressor_reduction_and_forward():
    model = DummyMLP()
    orig_params = sum(p.numel() for p in model.parameters())
    compressor = A3Compressor(model, rank=16, lora_rank=4, max_error=1.0)
    compressed, stats = compressor.compress()
    assert compressed is not None

    comp_params = sum(p.numel() for p in compressed.parameters())
    effective_ratio = orig_params / max(1, comp_params)
    assert effective_ratio > 1.05

    x = torch.randn(2, 128)
    out = compressed(x)
    assert out.shape == (2, 64)
    assert torch.isfinite(out).all()
