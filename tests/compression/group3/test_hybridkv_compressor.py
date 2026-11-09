#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit tests for HybridKVCompressor & HybridKVCache (Group 3, Key 43).
"""

import sys
if sys.path and (sys.path[0].endswith("/tests") or sys.path[0] == "tests"):
    sys.path.pop(0)

import pytest
import torch
import torch.nn as nn
from neuralops.compression.kv_cache.hybridkv_compressor import HybridKVCompressor, HybridKVCache
from neuralops.core.registry import COMPRESSOR_REGISTRY


class DummyMLP(nn.Module):
    def __init__(self, in_features: int = 128, out_features: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.fc2 = nn.Linear(out_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def test_hybridkv_registry():
    assert "hybridkv_compressor" in COMPRESSOR_REGISTRY
    assert COMPRESSOR_REGISTRY.get("hybridkv_compressor") is HybridKVCompressor


def test_hybridkv_cache_eviction():
    cache = HybridKVCache(max_cache_len=16, num_sinks=2, recent_window=4, budget=1.0)
    k = torch.randn(2, 4, 20, 32)
    v = torch.randn(2, 4, 20, 32)
    k_out, v_out = cache.update_and_compress(k, v)
    assert k_out.shape[2] == 16
    assert v_out.shape[2] == 16
    assert torch.isfinite(k_out).all()
    assert torch.isfinite(v_out).all()


def test_hybridkv_compress_and_forward():
    model = DummyMLP()
    compressor = HybridKVCompressor(model, bits=4, budget=0.5)
    compressed, stats = compressor.compress()

    assert compressed is not None
    assert getattr(compressed, "effective_bits", 32) <= 8

    x = torch.randn(2, 128)
    out = compressed(x)
    assert out.shape == (2, 128)
    assert torch.isfinite(out).all()
