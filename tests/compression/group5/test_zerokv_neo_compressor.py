#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit tests for ZeroKVNeoCompressor (Group 5, Key 82).
"""

import pytest
import torch
import torch.nn as nn
from neuralops.compression.kv_cache.zerokv_neo_compressor import (
    ZeroKVNeoCache,
    ZeroKVNeoCompressor,
    zerokv_neo_factory,
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
    assert "zerokv_neo_compressor" in COMPRESSOR_REGISTRY
    assert COMPRESSOR_REGISTRY.get("zerokv_neo_compressor") is not None


def test_zerokv_neo_cache_compaction():
    cache = ZeroKVNeoCache(max_cache_len=32, num_attention_sinks=4, recent_window=16)
    k = torch.randn(1, 4, 64, 32)
    v = torch.randn(1, 4, 64, 32)

    k_comp, v_comp = cache.update_and_compress(k, v)
    assert k_comp.size(2) <= 32
    assert v_comp.size(2) <= 32
    assert torch.isfinite(k_comp).all()
    assert torch.isfinite(v_comp).all()


def test_zerokv_neo_compressor_reduction():
    model = DummyMLP()
    compressor = ZeroKVNeoCompressor(model, keep_ratio=0.3, max_error=1.0)
    compressed, stats = compressor.compress()

    assert stats.get("effective_ratio", 1.0) > 1.05 or stats.get("sparsity", 0.0) >= 0.3

    x = torch.randn(2, 128)
    out = compressed(x)
    assert out.shape == (2, 64)
    assert torch.isfinite(out).all()


def test_zerokv_neo_factory():
    model = DummyMLP()
    comp_model, stats = zerokv_neo_factory(model, keep_ratio=0.3, return_stats=True)
    assert comp_model is not None
    assert isinstance(stats, dict)
