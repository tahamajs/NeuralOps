#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit tests for FAQQuantizer (Group 2, Key 30).
"""

import pytest
import torch
import torch.nn as nn
from neuralops.compression.quantization.faq_quantizer import (
    FAQLinear,
    FAQQuantizer,
    faq_factory,
)
from neuralops.core.registry import COMPRESSOR_REGISTRY


class DummyMLP(nn.Module):
    def __init__(self, in_features: int = 64, hidden: int = 128, out_features: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


def test_registry():
    assert "faq_quantizer" in COMPRESSOR_REGISTRY
    assert COMPRESSOR_REGISTRY.get("faq_quantizer") is not None


def test_faq_linear():
    lin = nn.Linear(64, 128)
    layer = FAQLinear.from_linear(lin, bits=3, lora_r=4)
    assert layer.bits == 3
    assert layer.in_features == 64
    assert layer.out_features == 128

    x = torch.randn(2, 64)
    out = layer(x)
    assert out.shape == (2, 128)
    assert torch.isfinite(out).all()


def test_faq_quantizer_reduction():
    model = DummyMLP()
    compressor = FAQQuantizer(model, bits=3)
    compressed, stats = compressor.compress()
    assert compressed is not None
    assert stats["bit_reduction"] > 1.05

    x = torch.randn(2, 64)
    out = compressed(x)
    assert out.shape == (2, 32)
    assert torch.isfinite(out).all()


def test_faq_factory():
    model = DummyMLP()
    compressed, stats = faq_factory(model, bits=3, return_stats=True)
    assert compressed is not None
    assert "bit_reduction" in stats
