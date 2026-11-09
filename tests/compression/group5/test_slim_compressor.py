#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit tests for SLiMCompressor (Group 5, Key 69).
"""

import pytest
import torch
import torch.nn as nn
from neuralops.compression.low_rank.slim_compressor import (
    SLiMCompressor,
    slim_factory,
    slim_compressor_factory,
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
    assert "slim_compressor" in COMPRESSOR_REGISTRY
    assert COMPRESSOR_REGISTRY.get("slim_compressor") is not None


def test_slim_compression_reduction():
    model = DummyMLP()
    compressor = SLiMCompressor(model, bits=4, sparsity=0.5, rank=8, max_error=1.0)
    compressed, stats = compressor.compress()

    # Verify bit reduction
    assert stats.get("bit_reduction", 1.0) >= 4.0 or stats.get("effective_parameter_ratio", 1.0) > 1.05

    # Check forward pass
    x = torch.randn(4, 128)
    out = compressed(x)
    assert out.shape == (4, 64)
    assert torch.isfinite(out).all()


def test_slim_mask_registration():
    model = DummyMLP()
    compressor = SLiMCompressor(model, bits=4, sparsity=0.5, rank=8)
    compressed, _ = compressor.compress()

    assert hasattr(compressed.fc1, "pruning_mask") or hasattr(compressed.fc1, "mask")


def test_slim_factory():
    model = DummyMLP()
    comp_model, stats = slim_factory(model, bits=4, return_stats=True)
    assert comp_model is not None
    assert isinstance(stats, dict)
