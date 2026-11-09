#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit tests for AWQ (Group 1, Key 13).
"""

import pytest
import torch
import torch.nn as nn
from neuralops.compression.quantization.awq import (
    AWQCompressor,
    AWQQuantizer,
    quantize_model_awq,
)
from neuralops.core.registry import COMPRESSOR_REGISTRY


def test_awq_registry():
    assert "awq" in COMPRESSOR_REGISTRY
    assert COMPRESSOR_REGISTRY.get("awq") is not None


def test_awq_numerical_preservation():
    torch.manual_seed(42)
    layer = nn.Linear(64, 64, bias=False)
    orig_w = layer.weight.data.clone()
    awq = AWQQuantizer(layer, bits=4, alpha=0.5)

    x = torch.randn(32, 64)
    x[:, 10] *= 20.0
    awq.observe(x)

    w_q = awq.quantize()
    assert not torch.equal(layer.weight.data, orig_w)

    with torch.no_grad():
        out_orig = torch.matmul(x, orig_w.t())
        out_q = layer(x)
        rel_err = torch.norm(out_orig - out_q) / torch.norm(out_orig)
        assert rel_err < 0.5


def test_quantize_model_awq():
    model = nn.Sequential(nn.Linear(32, 32), nn.Linear(32, 16))
    calib = [torch.randn(8, 32) for _ in range(2)]
    quantize_model_awq(model, calib, bits=4)
    out = model(torch.randn(2, 32))
    assert out.shape == (2, 16)
    assert torch.isfinite(out).all()


def test_awq_compressor_reduction_and_forward():
    model = nn.Sequential(nn.Linear(64, 128), nn.GELU(), nn.Linear(128, 32))
    compressor = AWQCompressor(model, bits=4)
    compressed, stats = compressor.compress()
    assert compressed is not None

    effective_ratio = float(stats.get("compression_ratio", 4.0))
    assert effective_ratio > 1.05

    x = torch.randn(2, 64)
    out = compressed(x)
    assert out.shape == (2, 32)
    assert torch.isfinite(out).all()
