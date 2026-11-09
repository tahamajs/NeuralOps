#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit tests for GPTQCompressor (Group 3, Key 37).
"""

import sys
if sys.path and (sys.path[0].endswith("/tests") or sys.path[0] == "tests"):
    sys.path.pop(0)

import pytest
import torch
import torch.nn as nn
from neuralops.compression.quantization.gptq import GPTQCompressor, GPTQQuantizer
from neuralops.core.registry import COMPRESSOR_REGISTRY


class DummyMLP(nn.Module):
    def __init__(self, in_features: int = 128, out_features: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.fc2 = nn.Linear(out_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def test_gptq_registry():
    assert "gptq" in COMPRESSOR_REGISTRY
    assert COMPRESSOR_REGISTRY.get("gptq") is GPTQCompressor


def test_gptq_quantizer_column():
    linear = nn.Linear(64, 64)
    quantizer = GPTQQuantizer(linear, bits=4, sym=True)
    x = torch.randn(10, 64)
    quantizer.add_batch(x)
    W_q = quantizer.quantize()
    assert W_q.shape == (64, 64)
    assert torch.isfinite(W_q).all()


def test_gptq_compress_and_forward():
    model = DummyMLP()
    compressor = GPTQCompressor(model, bits=4, sym=True)
    compressed, stats = compressor.compress()

    assert compressed is not None
    assert stats["bits"] == 4
    assert stats["compression_ratio"] >= 4.0

    x = torch.randn(2, 128)
    out = compressed(x)
    assert out.shape == (2, 128)
    assert torch.isfinite(out).all()
