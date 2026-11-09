#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit tests for Structure-Aware Low-Rank (SALR) Compressor (Group 4, Key 65).
"""

import pytest
import torch
import torch.nn as nn
from neuralops.compression.low_rank.salr import (
    SALRCompressor,
    SALRLayer,
    salr_factory,
)
from neuralops.core.registry import COMPRESSOR_REGISTRY


class DummyMLP(nn.Module):
    def __init__(self, in_features: int = 128, hidden: int = 256, out_features: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


def test_registry():
    assert "salr" in COMPRESSOR_REGISTRY
    assert COMPRESSOR_REGISTRY.get("salr") is not None


def test_salr_layer():
    layer = SALRLayer(128, 64, rank=16, sparsity=0.5)
    assert layer.in_features == 128
    assert layer.out_features == 64
    assert layer.rank == 16

    x = torch.randn(4, 128)
    out = layer(x)
    assert out.shape == (4, 64)
    assert torch.isfinite(out).all()


def test_salr_compressor_reduction():
    model = DummyMLP()
    orig_params = sum(p.numel() for p in model.parameters())
    compressor = SALRCompressor(
        model,
        rank=16,
        sparsity=0.5,
        max_error=1.0,
        fallback_to_original=False,
    )
    compressed, stats = compressor.compress()

    comp_params = sum(p.numel() for p in compressed.parameters())
    assert orig_params / max(comp_params, 1) > 1.05

    x = torch.randn(2, 128)
    out = compressed(x)
    assert out.shape == (2, 64)
    assert torch.isfinite(out).all()


def test_salr_factory():
    model = DummyMLP()
    comp_model, stats = salr_factory(
        model,
        rank=16,
        sparsity=0.5,
        max_error=1.0,
        fallback_to_original=False,
        return_stats=True,
    )
    assert comp_model is not None
    assert isinstance(stats, dict)
    x = torch.randn(2, 128)
    out = comp_model(x)
    assert out.shape == (2, 64)
    assert torch.isfinite(out).all()
