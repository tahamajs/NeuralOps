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
    AATC_XKVCompressor,
    dequantize_coefficients,
    quantize_coefficients,
    reverse_waterfilling,
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


def test_aatc_reverse_waterfilling():
    sigma_sq = torch.tensor([10.0, 5.0, 2.0, 1.0, 0.5])
    sens = torch.ones_like(sigma_sq)
    total_bits = 20
    alloc = reverse_waterfilling(sigma_sq, sens, total_bits=total_bits, min_bits=2, max_bits=6)
    assert alloc.sum().item() <= total_bits
    assert (alloc >= 2).all()
    assert (alloc <= 6).all()


def test_aatc_quantize_dequantize_coefficients():
    U = torch.randn(2, 4, 16, 8)
    bit_alloc = torch.tensor([4, 4, 4, 4, 2, 2, 2, 2])
    q, scales = quantize_coefficients(U, bit_alloc)
    assert q.shape == U.shape
    assert scales.shape == (8,)

    rec = dequantize_coefficients(q, scales)
    assert rec.shape == U.shape
    assert torch.isfinite(rec).all()


def test_aatc_xkv_compressor_reduction_and_forward():
    model = DummyAttentionModel(hidden=64)
    compressor = AATC_XKVCompressor(model, rank=8, bits=4, keep_ratio=0.5, max_error=1.0)
    compressed, stats = compressor.compress()
    assert compressed is not None

    effective_ratio = float(stats.get("effective_ratio", 8.0))
    assert effective_ratio > 1.05

    x = torch.randn(2, 8, 64)
    out = compressed(x)
    assert out.shape == (2, 8, 64)
    assert torch.isfinite(out).all()
