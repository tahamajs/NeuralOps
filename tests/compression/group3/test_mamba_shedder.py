#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit tests for MambaShedder & ShedLinear (Group 3, Key 51).
"""

import sys
if sys.path and (sys.path[0].endswith("/tests") or sys.path[0] == "tests"):
    sys.path.pop(0)

import pytest
import torch
import torch.nn as nn
from neuralops.compression.pruning.mamba_shedder import (
    MambaShedderCompressor,
    MambaShedderPrunedLinear,
    mamba_shedder_factory,
)
from neuralops.core.registry import COMPRESSOR_REGISTRY


class DummyMLP(nn.Module):
    def __init__(self, in_features: int = 128, out_features: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.fc2 = nn.Linear(out_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def test_mamba_shedder_registry():
    assert "mamba_shedder" in COMPRESSOR_REGISTRY
    assert COMPRESSOR_REGISTRY.get("mamba_shedder") is not None


def test_shed_linear_forward():
    W = torch.randn(48, 64)
    keep_idx = torch.arange(48)
    layer = MambaShedderPrunedLinear(in_features=64, original_out_features=64, W_kept=W, keep_idx=keep_idx)
    x = torch.randn(2, 64)
    out = layer(x)
    assert out.shape == (2, 64)
    assert torch.isfinite(out).all()


def test_mamba_shedder_compress_and_forward():
    model = DummyMLP()
    orig_params = sum(p.numel() for p in model.parameters())

    compressor = MambaShedderCompressor(
        model,
        sparsity=0.25,
        max_error=1.0,
        fallback_to_original=False,
        target_attention_only=False,
    )
    compressed, stats = compressor.compress()

    assert compressed is not None
    comp_params = sum(p.numel() for p in compressed.parameters())
    assert orig_params / comp_params > 1.05

    x = torch.randn(2, 128)
    out = compressed(x)
    assert out.shape == (2, 128)
    assert torch.isfinite(out).all()
