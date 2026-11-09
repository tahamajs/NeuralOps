#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit tests for AATC-XKV Compressor (Group 1, Key 3).
"""

import pytest
import torch
import torch.nn as nn
from neuralops.compression.kv_cache.aatc_xkv_compressor import (
    AATC_XKVCache,
    AATC_XKVCompressor,
    aatc_xkv_factory,
)
from neuralops.core.registry import COMPRESSOR_REGISTRY


class DummyAttentionModel(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.q_proj = nn.Linear(hidden, hidden)
        self.k_proj = nn.Linear(hidden, hidden)
        self.v_proj = nn.Linear(hidden, hidden)
        self.out_proj = nn.Linear(hidden, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        return self.out_proj(q + k + v)


def test_aatc_xkv_registry():
    assert "aatc_xkv_compressor" in COMPRESSOR_REGISTRY
    assert COMPRESSOR_REGISTRY.get("aatc_xkv_compressor") is not None


def test_aatc_kv_cache_operations():
    cache = AATCKVCache(rank=8, bits=4, sink_tokens=2, local_window=4)
    B, H, T, D = 2, 4, 16, 32
    k = torch.randn(B, H, T, D)
    v = torch.randn(B, H, T, D)

    cache.update_subspace(k, v)
    assert cache.V_k is not None
    assert cache.V_v is not None
    assert cache.V_k.shape == (8, D)

    k_c, v_c = cache.compress_kv(k, v)
    assert k_c.shape[2] == 16
    assert torch.isfinite(k_c).all()
    assert torch.isfinite(v_c).all()


def test_aatc_xkv_compressor_reduction_and_forward():
    model = DummyAttentionModel(hidden=64)
    orig_params = sum(p.numel() for p in model.parameters())

    compressor = AATCXKVCompressor(model, rank=8, bits=4, keep_ratio=0.5, max_error=1.0)
    compressed, stats = compressor.compress()
    assert compressed is not None

    effective_ratio = float(stats.get("effective_ratio", 8.0))
    assert effective_ratio > 1.05

    x = torch.randn(2, 8, 64)
    out = compressed(x)
    assert out.shape == (2, 8, 64)
    assert torch.isfinite(out).all()
