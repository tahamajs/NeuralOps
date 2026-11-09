#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit tests for HuffmanCoder & HuffmanCompressor (Group 3, Key 41).
"""

import sys
if sys.path and (sys.path[0].endswith("/tests") or sys.path[0] == "tests"):
    sys.path.pop(0)

import pytest
import torch
import torch.nn as nn
from neuralops.compression.quantization.huffman_coder import HuffmanCoder, HuffmanCompressor
from neuralops.core.registry import COMPRESSOR_REGISTRY


class DummyMLP(nn.Module):
    def __init__(self, in_features: int = 128, out_features: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.fc2 = nn.Linear(out_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def test_huffman_registry():
    assert "huffman_coder" in COMPRESSOR_REGISTRY
    assert COMPRESSOR_REGISTRY.get("huffman_coder") is HuffmanCompressor


def test_huffman_encode_decode():
    tensor = torch.randn(32, 32)
    packed_b, code_map, meta = HuffmanCoder.encode(tensor, quantize_bits=8)
    assert len(packed_b) > 0
    assert len(code_map) > 0
    decoded = HuffmanCoder.decode(packed_b, code_map, meta)
    assert decoded.shape == (32, 32)
    assert torch.allclose(tensor, decoded, atol=0.1)


def test_huffman_compress_and_forward():
    model = DummyMLP()
    compressor = HuffmanCompressor(model, quantize_bits=4, target_attention_only=False)
    compressed, stats = compressor.compress()

    assert compressed is not None
    eff_ratio = stats.get("huffman_compression_ratio", stats.get("theoretical_compression_ratio", 1.0))
    assert eff_ratio > 1.05

    x = torch.randn(2, 128)
    out = compressed(x)
    assert out.shape == (2, 128)
    assert torch.isfinite(out).all()
