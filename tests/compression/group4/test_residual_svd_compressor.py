#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit tests for Residual SVD Compressor (Group 4, Key 60).
"""

import pytest
import torch
import torch.nn as nn
from neuralops.compression.svd.residual_svd_compressor import (
    ResidualSVDCompressor,
    ResSVDCompressor,
    ResSVDLinear,
    res_svd_factory,
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
    assert "residual_svd_compressor" in COMPRESSOR_REGISTRY
    assert COMPRESSOR_REGISTRY.get("residual_svd_compressor") is not None


def test_res_svd_linear():
    lin = nn.Linear(128, 64)
    layer = ResSVDLinear.from_linear(lin, base_rank=16, residual_rank=4)
    assert layer.in_features == 128
    assert layer.out_features == 64
    assert layer.base_rank == 16
    assert layer.residual_rank == 4

    x = torch.randn(4, 128)
    out = layer(x)
    assert out.shape == (4, 64)
    assert torch.isfinite(out).all()


def test_residual_svd_compressor_reduction():
    model = DummyMLP()
    orig_params = sum(p.numel() for p in model.parameters())
    compressor = ResidualSVDCompressor(model, base_rank=16, residual_rank=4, factorize_base=True)
    compressed, stats = compressor.compress()

    comp_params = sum(p.numel() for p in compressed.parameters())
    assert orig_params / max(comp_params, 1) > 1.05

    x = torch.randn(2, 128)
    out = compressed(x)
    assert out.shape == (2, 64)
    assert torch.isfinite(out).all()


def test_res_svd_factory():
    model = DummyMLP()
    comp_model = res_svd_factory(model, base_rank=16, residual_rank=4)
    assert comp_model is not None
    x = torch.randn(2, 128)
    out = comp_model(x)
    assert out.shape == (2, 64)
    assert torch.isfinite(out).all()
