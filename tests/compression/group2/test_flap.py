#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit tests for FLAPCompressor (Group 2, Key 31).
"""

import pytest
import torch
import torch.nn as nn
from neuralops.compression.pruning.flap import (
    FLAPCompressor,
    flap_compressor_factory,
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
    assert "flap" in COMPRESSOR_REGISTRY
    assert COMPRESSOR_REGISTRY.get("flap") is not None


def test_flap_compressor_pruning():
    model = DummyMLP()
    orig_params = sum(p.numel() for p in model.parameters())
    compressor = FLAPCompressor(model, sparsity=0.2, max_error=1.0)
    compressed, stats = compressor.compress()
    assert compressed is not None
    eff_ratio = 1.0 / max(1e-6, stats.get("effective_parameter_ratio", 1.0))
    assert eff_ratio > 1.05 or stats.get("sparsity_target", 0.0) > 0.0

    x = torch.randn(2, 128)
    out = compressed(x)
    assert out.shape == (2, 64)
    assert torch.isfinite(out).all()


def test_flap_factory():
    model = DummyMLP()
    compressed = flap_compressor_factory(model, sparsity=0.2)
    assert compressed is not None
