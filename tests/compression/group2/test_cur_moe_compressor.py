#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit tests for CURMoECompressor (Group 2, Key 21).
"""

import pytest
import torch
import torch.nn as nn
from neuralops.compression.low_rank.cur_moe_compressor import (
    CURMoECompressor,
    cur_moe_factory,
)
from neuralops.core.registry import COMPRESSOR_REGISTRY


class DummyMoE(nn.Module):
    def __init__(self, d: int = 128, num_experts: int = 2):
        super().__init__()
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = 0.0
        for exp in self.experts:
            out = out + exp(x)
        return out


def test_registry():
    assert "cur_moe_compressor" in COMPRESSOR_REGISTRY
    assert COMPRESSOR_REGISTRY.get("cur_moe_compressor") is not None


def test_cur_moe_compressor_basic():
    model = DummyMoE(d=128)
    compressor = CURMoECompressor(model, rank=16, max_error=1.0)
    compressed, stats = compressor.compress()
    assert compressed is not None
    assert stats["effective_parameter_ratio"] > 1.05

    x = torch.randn(2, 4, 128)
    out = compressed(x)
    assert out.shape == (2, 4, 128)
    assert torch.isfinite(out).all()


def test_cur_moe_factory():
    model = DummyMoE(d=128)
    compressed, stats = cur_moe_factory(model, rank=16, return_stats=True)
    assert compressed is not None
    assert "effective_parameter_ratio" in stats
