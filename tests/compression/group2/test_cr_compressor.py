#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit tests for CRCompressor (Group 2, Key 19).
"""

import pytest
import torch
import torch.nn as nn
from neuralops.compression.low_rank.cr_compressor import (
    CRCompressor,
    CRLinear,
    cr_compressor_factory,
)
from neuralops.core.registry import COMPRESSOR_REGISTRY


class DummyMLP(nn.Module):
    def __init__(self, in_features: int = 128, hidden: int = 128, out_features: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


def test_registry():
    assert "cr_compressor" in COMPRESSOR_REGISTRY
    assert COMPRESSOR_REGISTRY.get("cr_compressor") is not None


def test_cr_linear_decomposition():
    lin = nn.Linear(128, 64)
    cr_layer = CRLinear.from_linear(lin, rank=16)
    assert cr_layer.rank == 16
    assert cr_layer.in_features == 128
    assert cr_layer.out_features == 64

    x = torch.randn(4, 128)
    out = cr_layer(x)
    assert out.shape == (4, 64)
    assert torch.isfinite(out).all()

    w_rec = cr_layer.get_weight()
    assert w_rec.shape == (64, 128)
    err = torch.norm(lin.weight - w_rec) / torch.norm(lin.weight)
    assert err < 1.0


def test_cr_compressor_reduction():
    model = DummyMLP()
    compressor = CRCompressor(model, rank=16, max_error=1.0)
    compressed, stats = compressor.compress()
    assert compressed is not None
    assert stats["effective_parameter_ratio"] > 1.05

    x = torch.randn(2, 128)
    out = compressed(x)
    assert out.shape == (2, 64)
    assert torch.isfinite(out).all()


def test_cr_factory():
    model = DummyMLP()
    compressed, stats = cr_compressor_factory(model, rank=16, return_stats=True)
    assert compressed is not None
    assert "effective_parameter_ratio" in stats
