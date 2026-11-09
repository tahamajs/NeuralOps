#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit tests for SAES-SVD v2 Compressor (Group 4, Key 64).
"""

import pytest
import torch
import torch.nn as nn
from neuralops.compression.svd.saes_svd_v2 import (
    SAESSVDCompressorV2,
    SAESCompressor,
    saes_svd_v2_factory,
)
from neuralops.core.registry import COMPRESSOR_REGISTRY


class DummyMLP(nn.Module):
    def __init__(self, in_features: int = 128, hidden: int = 256, out_features: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


def test_registry():
    assert "saes_svd_v2" in COMPRESSOR_REGISTRY
    assert COMPRESSOR_REGISTRY.get("saes_svd_v2") is not None
    assert "saes" in COMPRESSOR_REGISTRY


def test_saes_svd_v2_reduction():
    model = DummyMLP()
    orig_params = sum(p.numel() for p in model.parameters())
    compressor = SAESSVDCompressorV2(
        model,
        rank=16,
        rank_ratio=0.25,
        fallback_to_original=False,
    )
    compressed, stats = compressor.compress()

    comp_params = sum(p.numel() for p in compressed.parameters())
    assert orig_params / max(comp_params, 1) > 1.05

    x = torch.randn(2, 128)
    out = compressed(x)
    assert out.shape == (2, 64)
    assert torch.isfinite(out).all()


def test_saes_svd_v2_factory():
    model = DummyMLP()
    comp_model, stats = saes_svd_v2_factory(
        model,
        rank=16,
        rank_ratio=0.25,
        fallback_to_original=False,
        return_stats=True,
    )
    assert comp_model is not None
    assert isinstance(stats, dict)
    x = torch.randn(2, 128)
    out = comp_model(x)
    assert out.shape == (2, 64)
    assert torch.isfinite(out).all()
