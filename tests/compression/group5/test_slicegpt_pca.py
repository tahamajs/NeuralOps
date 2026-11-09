#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit tests for SliceGPTPCA (Group 5, Key 68).
"""

import pytest
import torch
import torch.nn as nn
from neuralops.compression.quantization.slicegpt_pca import (
    SliceGPTPCACompressor,
    SliceGPTPCALinear,
    slicegpt_pca_factory,
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
    assert "slicegpt_pca" in COMPRESSOR_REGISTRY
    assert COMPRESSOR_REGISTRY.get("slicegpt_pca") is not None


def test_slicegpt_pca_linear():
    lin = nn.Linear(128, 64)
    sliced_w = lin.weight.data[:, :96].contiguous()
    layer = SliceGPTPCALinear.from_linear(lin, sliced_weight=sliced_w)
    assert layer.in_features == 96
    assert layer.out_features == 64

    x = torch.randn(4, 96)
    out = layer(x)
    assert out.shape == (4, 64)
    assert torch.isfinite(out).all()


def test_slicegpt_pca_compressor_reduction():
    model = DummyMLP()
    orig_params = sum(p.numel() for p in model.parameters())
    compressor = SliceGPTPCACompressor(model, sparsity=0.5, pca_ratio=0.5)
    compressed, stats = compressor.compress()

    comp_params = sum(p.numel() for p in compressed.parameters())
    assert orig_params / max(comp_params, 1) > 1.05

    x = torch.randn(2, 128)
    out = compressed(x)
    assert out.shape == (2, 64)
    assert torch.isfinite(out).all()


def test_slicegpt_pca_factory():
    model = DummyMLP()
    comp_model, stats = slicegpt_pca_factory(model, sparsity=0.3, return_stats=True)
    assert comp_model is not None
    assert isinstance(stats, dict)
