#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit tests for MGIESVDBudgetCompressor (Group 4, Key 52: mgie_budget).
"""

import pytest
import torch
import torch.nn as nn
from neuralops.compression.low_rank.mgie_svd_compressor import (
    MGIESVDBudgetCompressor,
    mgie_budget_factory,
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
    assert "mgie_budget" in COMPRESSOR_REGISTRY
    assert COMPRESSOR_REGISTRY.get("mgie_budget") is not None


def test_mgie_budget_compression():
    model = DummyMLP()
    comp = MGIESVDBudgetCompressor(model, rank=16, total_rank_budget=64)
    compressed, stats = comp.compress()
    assert compressed is not None
    assert stats["forward_check_success"] is True

    x = torch.randn(2, 128)
    out = compressed(x)
    assert out.shape == (2, 128)
    assert torch.isfinite(out).all()


def test_mgie_budget_factory():
    model = DummyMLP()
    compressed = mgie_budget_factory(model, rank=16)
    assert compressed is not None
    x = torch.randn(2, 128)
    out = compressed(x)
    assert torch.isfinite(out).all()
