#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit tests for DeltaKVCompressor (Group 2, Key 22).
"""

import pytest
import torch
import torch.nn as nn
from neuralops.compression.kv_cache.delta_kv import (
    DeltaKVCompressor,
    DeltaKVLinear,
    delta_kv_factory,
)
from neuralops.core.registry import COMPRESSOR_REGISTRY


class DummyKVModel(nn.Module):
    def __init__(self, d: int = 128):
        super().__init__()
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k = self.k_proj(x)
        v = self.v_proj(x)
        return self.out_proj(k + v)


def test_registry():
    assert "delta_kv" in COMPRESSOR_REGISTRY
    assert COMPRESSOR_REGISTRY.get("delta_kv") is not None


def test_delta_kv_linear():
    layer = DeltaKVLinear(in_features=128, out_features=128, rank=16)
    assert layer.down.weight.shape == (16, 128)
    assert layer.up.weight.shape == (128, 16)

    x = torch.randn(2, 8, 128)
    out = layer(x)
    assert out.shape == (2, 8, 128)
    assert torch.isfinite(out).all()


def test_delta_kv_compressor_reduction():
    model = DummyKVModel(d=128)
    orig_params = sum(p.numel() for p in model.parameters())
    compressor = DeltaKVCompressor(model, rank=16)
    compressed, stats = compressor.compress()
    assert compressed is not None
    comp_params = sum(p.numel() for p in compressed.parameters())
    assert (orig_params / comp_params) > 1.05

    x = torch.randn(2, 4, 128)
    out = compressed(x)
    assert out.shape == (2, 4, 128)
    assert torch.isfinite(out).all()


def test_delta_kv_factory():
    model = DummyKVModel(d=128)
    compressed = delta_kv_factory(model, rank=16)
    assert compressed is not None
