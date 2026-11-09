#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit tests for KroneckerCompressor & KroneckerLinear (Group 3, Key 45).
"""

import sys
if sys.path and (sys.path[0].endswith("/tests") or sys.path[0] == "tests"):
    sys.path.pop(0)

import pytest
import torch
import torch.nn as nn
from neuralops.compression.tensor_decomp.kronecker_compressor import (
    KroneckerCompressor,
    KroneckerLinear,
    factorize_dim,
)
from neuralops.core.registry import COMPRESSOR_REGISTRY


class DummyMLP(nn.Module):
    def __init__(self, in_features: int = 128, out_features: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.fc2 = nn.Linear(out_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def test_kronecker_registry():
    assert "kronecker_compressor" in COMPRESSOR_REGISTRY
    assert COMPRESSOR_REGISTRY.get("kronecker_compressor") is not None


def test_factorize_dim():
    assert factorize_dim(128) == (8, 16) or factorize_dim(128) == (11, 11) or (factorize_dim(128)[0] * factorize_dim(128)[1] == 128)
    a, b = factorize_dim(64)
    assert a * b == 64


def test_kronecker_linear_forward():
    k_lin = KroneckerLinear(in_features=64, out_features=64, m1=8, n1=8, m2=8, n2=8)
    x = torch.randn(2, 64)
    out = k_lin(x)
    assert out.shape == (2, 64)
    assert torch.isfinite(out).all()


def test_kronecker_compress_and_forward():
    model = DummyMLP()
    orig_params = sum(p.numel() for p in model.parameters())

    compressor = KroneckerCompressor(model, rank=16, max_error=1.0, fallback_to_original=False)
    compressed, stats = compressor.compress()

    assert compressed is not None
    comp_params = sum(p.numel() for p in compressed.parameters())
    assert orig_params / comp_params > 1.05

    x = torch.randn(2, 128)
    out = compressed(x)
    assert out.shape == (2, 128)
    assert torch.isfinite(out).all()
