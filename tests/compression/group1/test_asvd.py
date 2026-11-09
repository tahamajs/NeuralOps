#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit tests for ASVDCompressor and ASVD Attention (Group 1, Key 11).
"""

import pytest
import torch
import torch.nn as nn
from neuralops.core.registry import COMPRESSOR_REGISTRY
from tests.testing_utils import DummyDistilBertAttention, MockConfig


class DummyMLP(nn.Module):
    def __init__(self, in_features: int = 128, hidden: int = 256, out_features: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


def test_asvd_registry():
    assert "asvd" in COMPRESSOR_REGISTRY
    assert COMPRESSOR_REGISTRY.get("asvd") is not None


def test_asvd_compressor_reduction_and_forward():
    cls = COMPRESSOR_REGISTRY.get("asvd")
    model = DummyMLP()
    orig_params = sum(p.numel() for p in model.parameters())

    compressor = cls(model, rank=16, max_error=1.0)
    compressed, stats = compressor.compress()
    assert compressed is not None

    comp_params = sum(p.numel() for p in compressed.parameters())
    effective_ratio = orig_params / max(1, comp_params)
    assert effective_ratio > 1.05

    x = torch.randn(2, 128)
    out = compressed(x)
    assert out.shape == (2, 64)
    assert torch.isfinite(out).all()


def test_asvd_attention_basic():
    try:
        from neuralops.attentions.asvd_attention import ASVDAttention

        cfg = MockConfig()
        attn = ASVDAttention(config=cfg)
        x = torch.randn(2, 8, 64)
        out, weights = attn(x, output_attentions=True)
        assert out.shape == (2, 8, 64)
        assert torch.isfinite(out).all()
    except ImportError:
        pass
