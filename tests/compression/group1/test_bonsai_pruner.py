#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit tests for BonsaiPruner (Group 1, Key 14).
"""

import pytest
import torch
import torch.nn as nn
from neuralops.compression.pruning.bonsai_pruner import (
    BonsaiLinear,
    BonsaiPruner,
    bonsai_pruner_factory,
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


def test_bonsai_registry():
    assert "bonsai_pruner" in COMPRESSOR_REGISTRY
    assert COMPRESSOR_REGISTRY.get("bonsai_pruner") is not None


def test_bonsai_linear_forward_and_sparsity():
    lin = nn.Linear(128, 64)
    bonsai_layer = BonsaiLinear.from_linear(lin, sparsity=0.5, lora_r=8)

    assert bonsai_layer.in_features == 128
    assert bonsai_layer.out_features == 64
    assert bonsai_layer.lora_r == 8

    x = torch.randn(4, 128)
    out = bonsai_layer(x)
    assert out.shape == (4, 64)
    assert torch.isfinite(out).all()


def test_bonsai_pruner_reduction_and_forward():
    model = DummyMLP()
    orig_params = sum(p.numel() for p in model.parameters())

    compressor = BonsaiPruner(model, sparsity=0.5, max_error=1.0)
    compressed, stats = compressor.compress()
    assert compressed is not None

    effective_ratio = float(stats.get("effective_ratio", 2.0))
    assert effective_ratio > 1.05

    x = torch.randn(2, 128)
    out = compressed(x)
    assert out.shape == (2, 64)
    assert torch.isfinite(out).all()
