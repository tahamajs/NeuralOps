#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit tests for COMPOTCompressor (Group 1, Key 17).
"""

import pytest
import torch
import torch.nn as nn
from neuralops.compression.low_rank.compot_compressor import (
    COMPOTCompressor,
    COMPOTLinear,
    compot_compressor_factory,
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


def test_compot_registry():
    assert "compot_compressor" in COMPRESSOR_REGISTRY
    assert COMPRESSOR_REGISTRY.get("compot_compressor") is not None


def test_compot_linear_forward_and_weight():
    lin = nn.Linear(128, 64)
    compot_layer = COMPOTLinear.from_linear(lin, rank=16)

    assert compot_layer.in_features == 128
    assert compot_layer.out_features == 64
    assert compot_layer.rank == 16

    x = torch.randn(4, 128)
    out = compot_layer(x)
    assert out.shape == (4, 64)
    assert torch.isfinite(out).all()

    w = compot_layer.get_weight()
    assert w.shape == (64, 128)


def test_compot_compressor_reduction_and_forward():
    model = DummyMLP()
    orig_params = sum(p.numel() for p in model.parameters())

    compressor = COMPOTCompressor(model, rank=16, max_error=1.0)
    compressed, stats = compressor.compress()
    assert compressed is not None

    comp_params = sum(p.numel() for p in compressed.parameters())
    effective_ratio = orig_params / max(1, comp_params)
    assert effective_ratio > 1.05

    x = torch.randn(2, 128)
    out = compressed(x)
    assert out.shape == (2, 64)
    assert torch.isfinite(out).all()
