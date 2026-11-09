#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit tests for QuaRotCompressor (Group 4, Key 59: quarot).
"""

import pytest
import torch
import torch.nn as nn
from neuralops.compression.quantization.quarot import (
    QuaRotCompressor,
    quarot_factory,
)
from neuralops.core.registry import COMPRESSOR_REGISTRY
from neuralops.ops.triton.elementwise import fast_hadamard_transform


class DummyMLP(nn.Module):
    def __init__(self, in_features: int = 128, out_features: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(out_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


def test_registry():
    assert "quarot" in COMPRESSOR_REGISTRY
    assert COMPRESSOR_REGISTRY.get("quarot") is not None


def test_hadamard_transform_shape():
    w = torch.randn(64, 128)
    rot = fast_hadamard_transform(w, block_size=64)
    assert rot.shape == w.shape
    assert torch.isfinite(rot).all()


def test_quarot_compressor():
    model = DummyMLP()
    comp = QuaRotCompressor(model, bits=4, hadamard_block_size=64)
    compressed, stats = comp.compress()
    assert compressed is not None
    assert stats["bits"] == 4
    assert stats["forward_check_success"] is True

    x = torch.randn(2, 128)
    out = compressed(x)
    assert out.shape == (2, 128)
    assert torch.isfinite(out).all()


def test_quarot_factory():
    model = DummyMLP()
    compressed = quarot_factory(model, bits=4, hadamard_block_size=64)
    assert compressed is not None
    x = torch.randn(2, 128)
    out = compressed(x)
    assert torch.isfinite(out).all()
