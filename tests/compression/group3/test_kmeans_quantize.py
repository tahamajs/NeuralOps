#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit tests for KMeansCompressor & fast_kmeans_1d (Group 3, Key 44).
"""

import sys
if sys.path and (sys.path[0].endswith("/tests") or sys.path[0] == "tests"):
    sys.path.pop(0)

import pytest
import torch
import torch.nn as nn
from neuralops.compression.quantization.kmeans_quantize import (
    KMeansCompressor,
    KMeansLinear,
    fast_kmeans_1d,
)
from neuralops.core.registry import COMPRESSOR_REGISTRY


class DummyMLP(nn.Module):
    def __init__(self, in_features: int = 128, out_features: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.fc2 = nn.Linear(out_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def test_kmeans_registry():
    assert "kmeans_quantize" in COMPRESSOR_REGISTRY
    assert COMPRESSOR_REGISTRY.get("kmeans_quantize") is not None


def test_fast_kmeans_1d():
    weights = torch.randn(64, 64)
    centroids, indices = fast_kmeans_1d(weights, n_clusters=16)
    assert centroids.numel() == 16
    assert indices.shape == (64, 64)
    assert (indices >= 0).all() and (indices < 16).all()


def test_kmeans_linear_layer():
    linear = nn.Linear(64, 64)
    km_layer = KMeansLinear(linear, n_clusters=16)
    x = torch.randn(2, 64)
    out = km_layer(x)
    assert out.shape == (2, 64)
    assert torch.isfinite(out).all()


def test_kmeans_compress_and_forward():
    model = DummyMLP()
    orig_params = sum(p.numel() for p in model.parameters())

    compressor = KMeansCompressor(model, n_clusters=16)
    compressed, stats = compressor.compress()

    assert compressed is not None
    comp_params = sum(p.numel() for p in compressed.parameters())
    assert orig_params / comp_params > 1.05

    x = torch.randn(2, 128)
    out = compressed(x)
    assert out.shape == (2, 128)
    assert torch.isfinite(out).all()
