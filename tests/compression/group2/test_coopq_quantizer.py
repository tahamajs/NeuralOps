#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit tests for CoopQQuantizer (Group 2, Key 18).
"""

import pytest
import torch
import torch.nn as nn
from neuralops.compression.quantization.coopq_quantizer import (
    CoopQLinear,
    CoopQQuantizer,
    coopq_quantizer_factory,
    quantize_dequantize_coopq,
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
    assert "coopq_quantizer" in COMPRESSOR_REGISTRY
    assert COMPRESSOR_REGISTRY.get("coopq_quantizer") is not None


def test_coopq_quantize_dequantize_bounds():
    W = torch.randn(64, 128)
    W_q = quantize_dequantize_coopq(W, bits=4, symmetric=False, per_channel=True)
    assert W_q.shape == W.shape
    assert not torch.isnan(W_q).any()
    assert not torch.isinf(W_q).any()
    err = torch.norm(W - W_q) / torch.norm(W)
    assert err < 0.5


def test_coopq_linear_from_linear():
    lin = nn.Linear(64, 128)
    q_lin = CoopQLinear.from_linear(lin, bits=4, symmetric=False, lora_r=4)
    assert q_lin.bits == 4
    assert q_lin.in_features == 64
    assert q_lin.out_features == 128
    assert q_lin.lora_A is not None
    assert q_lin.lora_B is not None

    x = torch.randn(2, 64)
    out = q_lin(x)
    assert out.shape == (2, 128)
    assert torch.isfinite(out).all()


def test_coopq_compressor_full():
    model = DummyMLP()
    compressor = CoopQQuantizer(model, bits=4)
    compressed, stats = compressor.compress()
    assert compressed is not None
    assert stats["bit_reduction"] >= 4.0

    x = torch.randn(4, 64)
    out = compressed(x)
    assert out.shape == (4, 32)
    assert torch.isfinite(out).all()


def test_coopq_factory():
    model = DummyMLP()
    compressed, stats = coopq_quantizer_factory(model, bits=4, return_stats=True)
    assert compressed is not None
    assert "bit_reduction" in stats
