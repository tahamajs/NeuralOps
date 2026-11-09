#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit tests for TTCompressor (Group 5, Key 79).
"""

import pytest
import torch
import torch.nn as nn
from neuralops.compression.tensor_decomp.tt_compressor import (
    TTCompressor,
    TTLinearLayer,
    tt_compressor_factory,
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
    assert "tt_compressor" in COMPRESSOR_REGISTRY
    assert COMPRESSOR_REGISTRY.get("tt_compressor") is not None


def test_tt_compressor_reduction():
    model = DummyMLP()
    orig_params = sum(p.numel() for p in model.parameters())
    compressor = TTCompressor(model, rank=8, max_error=1.0)
    compressed, stats = compressor.compress()

    comp_params = sum(p.numel() for p in compressed.parameters())
    assert orig_params / max(comp_params, 1) > 1.05

    x = torch.randn(2, 128)
    out = compressed(x)
    assert out.shape == (2, 64)
    assert torch.isfinite(out).all()


def test_tt_factory():
    model = DummyMLP()
    comp_model, stats = tt_compressor_factory(model, rank=8, return_stats=True)
    assert comp_model is not None
    assert isinstance(stats, dict)
