#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit tests for KVTCCompressor & KVTCQuantizedLoRALinear (Group 3, Key 47).
"""

import sys
if sys.path and (sys.path[0].endswith("/tests") or sys.path[0] == "tests"):
    sys.path.pop(0)

import pytest
import torch
import torch.nn as nn
from neuralops.compression.kv_cache.kvtc_compressor import (
    KVTCCompressor,
    KVTCQuantizedLoRALinear,
    SimpleEntropyCoder,
)
from neuralops.core.registry import COMPRESSOR_REGISTRY


class DummyMLP(nn.Module):
    def __init__(self, in_features: int = 128, out_features: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.fc2 = nn.Linear(out_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def test_kvtc_registry():
    assert "kvtc_compressor" in COMPRESSOR_REGISTRY
    assert COMPRESSOR_REGISTRY.get("kvtc_compressor") is KVTCCompressor


def test_simple_entropy_coder():
    t = torch.randint(0, 4, (100,))
    ent = SimpleEntropyCoder.compute_entropy(t)
    assert 0.0 <= ent <= 2.5
    ratio = SimpleEntropyCoder.compute_compression_ratio(t, uncompressed_bits=16)
    assert ratio > 1.0


def test_kvtc_quantized_linear_forward():
    int_w = torch.randint(-8, 7, (64, 64), dtype=torch.int8)
    scale = torch.tensor(0.05)
    zero_p = torch.tensor(0.0)
    layer = KVTCQuantizedLoRALinear(64, 64, int_w, scale, zero_p)
    x = torch.randn(2, 64)
    out = layer(x)
    assert out.shape == (2, 64)
    assert torch.isfinite(out).all()


def test_kvtc_compress_and_forward():
    model = DummyMLP()
    compressor = KVTCCompressor(model, bits=4, max_error=1.0, fallback_to_original=False)
    compressed, stats = compressor.compress()

    assert compressed is not None
    assert stats["bits"] == 4

    x = torch.randn(2, 128)
    out = compressed(x)
    assert out.shape == (2, 128)
    assert torch.isfinite(out).all()
