#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit tests for OptimalSubspaceProjectionCompressor (Group 4, Key 57: optimal_subspace_projection).
"""

import pytest
import torch
import torch.nn as nn
from neuralops.compression.svd.optimal_subspace_projection import (
    OptimalSubspaceProjectionCompressor,
    optimal_subspace_projection_factory,
    OptimalSubspaceLoRALinear,
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
    assert "optimal_subspace_projection" in COMPRESSOR_REGISTRY
    assert COMPRESSOR_REGISTRY.get("optimal_subspace_projection") is not None


def test_osp_linear_forward():
    lin = nn.Linear(128, 128)
    U = torch.randn(128, 16)
    V = torch.randn(16, 128)
    layer = OptimalSubspaceLoRALinear(128, 128, U=U, V=V, bias=lin.bias)
    x = torch.randn(2, 128)
    out = layer(x)
    assert out.shape == (2, 128)
    assert torch.isfinite(out).all()


def test_osp_compressor():
    model = DummyMLP()
    comp = OptimalSubspaceProjectionCompressor(model, rank=16)
    compressed, stats = comp.compress()
    assert compressed is not None
    assert stats["forward_check_success"] is True

    x = torch.randn(2, 128)
    out = compressed(x)
    assert out.shape == (2, 128)
    assert torch.isfinite(out).all()


def test_osp_factory():
    model = DummyMLP()
    compressed = optimal_subspace_projection_factory(model, rank=16)
    assert compressed is not None
    x = torch.randn(2, 128)
    out = compressed(x)
    assert torch.isfinite(out).all()
