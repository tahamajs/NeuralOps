#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit tests for ProxSparseCompressor (Group 4, Key 58: proxsparse).
"""

import pytest
import torch
import torch.nn as nn
from neuralops.compression.pruning.proxsparse import (
    ProxSparseCompressor,
    proxsparse_factory,
    ProxSparsePrunedLoRALinear,
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
    assert "proxsparse" in COMPRESSOR_REGISTRY
    assert COMPRESSOR_REGISTRY.get("proxsparse") is not None


def test_proxsparse_linear_forward():
    lin = nn.Linear(128, 128)
    mask = torch.ones(128, 128)
    mask[:, :64] = 0.0
    pruned_w = lin.weight.data * mask
    layer = ProxSparsePrunedLoRALinear(128, 128, pruned_weight=pruned_w, mask=mask, bias=lin.bias)
    x = torch.randn(2, 128)
    out = layer(x)
    assert out.shape == (2, 128)
    assert torch.isfinite(out).all()


def test_proxsparse_compressor():
    model = DummyMLP()
    comp = ProxSparseCompressor(model, sparsity=0.5)
    compressed, stats = comp.compress()
    assert compressed is not None
    assert stats["forward_check_success"] is True

    x = torch.randn(2, 128)
    out = compressed(x)
    assert out.shape == (2, 128)
    assert torch.isfinite(out).all()


def test_proxsparse_factory():
    model = DummyMLP()
    compressed = proxsparse_factory(model, sparsity=0.5)
    assert compressed is not None
    x = torch.randn(2, 128)
    out = compressed(x)
    assert torch.isfinite(out).all()
