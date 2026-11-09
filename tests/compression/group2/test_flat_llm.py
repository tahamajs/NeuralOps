#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit tests for FlatLLMCompressor (Group 2, Key 32).
"""

import pytest
import torch
import torch.nn as nn
from neuralops.compression.experimental.flat_llm import (
    FLATLLMCompressor,
    PrunedChannelLinear,
    flat_llm_factory,
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
    assert "flat_llm" in COMPRESSOR_REGISTRY
    assert COMPRESSOR_REGISTRY.get("flat_llm") is not None


def test_pruned_channel_linear():
    W_kept = torch.randn(32, 128)
    kept_idx = torch.arange(32)
    layer = PrunedChannelLinear(in_features=128, original_out_features=64, kept_idx=kept_idx, W_kept=W_kept)
    assert layer.num_kept == 32
    assert layer.original_out_features == 64

    x = torch.randn(2, 128)
    out = layer(x)
    assert out.shape == (2, 64)
    assert torch.isfinite(out).all()


def test_flat_llm_compressor_reduction():
    model = DummyMLP()
    orig_params = sum(p.numel() for p in model.parameters())
    compressor = FLATLLMCompressor(model, prune_ratio=0.3, max_error=1.0)
    compressed, stats = compressor.compress()
    assert compressed is not None
    comp_params = sum(p.numel() for p in compressed.parameters())
    assert (orig_params / comp_params) > 1.05

    x = torch.randn(2, 128)
    out = compressed(x)
    assert out.shape == (2, 64)
    assert torch.isfinite(out).all()


def test_flat_llm_factory():
    model = DummyMLP()
    compressed = flat_llm_factory(model, prune_ratio=0.3)
    assert compressed is not None
