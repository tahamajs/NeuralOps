#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit tests for COMECompressor (Group 1, Key 16).
"""

import pytest
import torch
import torch.nn as nn
from neuralops.compression.low_rank.come_compressor import (
    COMEBridge,
    COMECompressor,
    COMELoRALinear,
    come_compressor_factory,
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


def test_come_registry():
    assert "come_compressor" in COMPRESSOR_REGISTRY
    assert COMPRESSOR_REGISTRY.get("come_compressor") is not None


def test_come_linear_forward():
    lin = nn.Linear(128, 64)
    come_layer = COMELoRALinear.from_linear(lin, lora_r=8)

    assert come_layer.in_features == 128
    assert come_layer.out_features == 64
    assert come_layer.lora_r == 8

    x = torch.randn(4, 128)
    out = come_layer(x)
    assert out.shape == (4, 64)
    assert torch.isfinite(out).all()


def test_come_bridge_forward():
    bridge = COMEBridge(hidden_size=64, rank=16)
    x = torch.randn(2, 10, 64)
    out = bridge(x)
    assert out.shape == (2, 10, 64)
    assert torch.isfinite(out).all()


def test_come_compressor_reduction_and_forward():
    model = DummyMLP()
    orig_params = sum(p.numel() for p in model.parameters())

    compressor = COMECompressor(model, rank=16, lora_r=4, max_error=1.0)
    compressed, stats = compressor.compress()
    assert compressed is not None

    comp_params = sum(p.numel() for p in compressed.parameters())
    effective_ratio = orig_params / max(1, comp_params)
    assert effective_ratio > 1.05

    x = torch.randn(2, 128)
    out = compressed(x)
    assert out.shape == (2, 64)
    assert torch.isfinite(out).all()
