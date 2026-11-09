#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit tests for DirectSVDCompressor (Group 2, Key 23).
"""

import pytest
import torch
import torch.nn as nn
from neuralops.compression.svd.direct_svd import (
    DirectSVDCompressor,
    direct_svd_factory,
)
from neuralops.core.registry import COMPRESSOR_REGISTRY
from neuralops.ops.cuda.direct_svd import direct_svd_compress, DirectSVDFunction


class DummyModel(nn.Module):
    def __init__(self, in_features: int = 128, out_features: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.fc2 = nn.Linear(out_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def test_registry():
    assert "direct_svd" in COMPRESSOR_REGISTRY
    assert COMPRESSOR_REGISTRY.get("direct_svd") is not None


def test_direct_svd_kernel_fallback():
    U = torch.randn(64, 16)
    V = torch.randn(16, 64)
    out = DirectSVDFunction.apply(U, V, 1.0)
    assert out.shape == (64, 64)
    assert torch.isfinite(out).all()
    assert torch.allclose(out, U @ V, atol=1e-4)


def test_direct_svd_compress_tensor():
    W = torch.randn(64, 64)
    W_rec = direct_svd_compress(W, rank=16)
    assert W_rec.shape == W.shape
    assert torch.isfinite(W_rec).all()


def test_direct_svd_compressor_reduction():
    model = DummyModel()
    orig_params = sum(p.numel() for p in model.parameters())
    compressor = DirectSVDCompressor(model, rank=16, max_error=1.0)
    compressed, stats = compressor.compress()
    assert compressed is not None
    comp_params = sum(p.numel() for p in compressed.parameters())
    assert (orig_params / comp_params) > 1.05

    x = torch.randn(2, 128)
    out = compressed(x)
    assert out.shape == (2, 128)
    assert torch.isfinite(out).all()


def test_direct_svd_factory():
    model = DummyModel()
    compressed = direct_svd_factory(model, rank=16, max_error=1.0)
    assert compressed is not None
