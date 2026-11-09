#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit tests for HashedNetCompressor (Group 3, Key 40).
"""

import sys
if sys.path and (sys.path[0].endswith("/tests") or sys.path[0] == "tests"):
    sys.path.pop(0)

import pytest
import torch
import torch.nn as nn
from neuralops.compression.experimental.hashed_net_compressor import (
    HashedNetCompressor,
    HashedLinear,
    fast_hash_and_sign,
)
from neuralops.core.registry import COMPRESSOR_REGISTRY


class DummyMLP(nn.Module):
    def __init__(self, in_features: int = 128, out_features: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.fc2 = nn.Linear(out_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def test_hashed_net_registry():
    assert "hashed_net_compressor" in COMPRESSOR_REGISTRY
    assert COMPRESSOR_REGISTRY.get("hashed_net_compressor") is HashedNetCompressor


def test_fast_hash_and_sign():
    hash_idx, sign_val = fast_hash_and_sign(out_features=64, in_features=64, n_buckets=256)
    assert hash_idx.shape == (64, 64)
    assert sign_val.shape == (64, 64)
    assert (hash_idx >= 0).all() and (hash_idx < 256).all()
    assert ((sign_val == 1) | (sign_val == -1)).all()


def test_hashed_linear_layer():
    linear = HashedLinear(in_features=64, out_features=64, n_buckets=512)
    x = torch.randn(2, 64)
    out = linear(x)
    assert out.shape == (2, 64)
    assert torch.isfinite(out).all()


def test_hashed_net_compress_and_forward():
    model = DummyMLP()
    orig_params = sum(p.numel() for p in model.parameters())

    compressor = HashedNetCompressor(model, n_buckets=1024)
    compressed, stats = compressor.compress()

    assert compressed is not None
    comp_params = sum(p.numel() for p in compressed.parameters())
    assert orig_params / comp_params > 1.05

    x = torch.randn(2, 128)
    out = compressed(x)
    assert out.shape == (2, 128)
    assert torch.isfinite(out).all()
