#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit tests for Rotational SliceGPT INT4 (Group 4, Key 62).
"""

import pytest
import torch
import torch.nn as nn
from neuralops.compression.quantization.rotational_slicegpt_int4 import (
    RotationalSliceGPTINT4Compressor,
    RotationalSliceGPTINT4Linear,
    rotational_slicegpt_int4_factory,
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
    assert "rotational_slicegpt_int4" in COMPRESSOR_REGISTRY
    assert COMPRESSOR_REGISTRY.get("rotational_slicegpt_int4") is not None


def test_rotational_slicegpt_int4_linear():
    layer = RotationalSliceGPTINT4Linear(128, 64, r=16, quant_bits=4)
    assert layer.in_features == 128
    assert layer.out_features == 64
    assert layer.r == 16
    assert layer.quant_bits == 4

    x = torch.randn(4, 128)
    out = layer(x)
    assert out.shape == (4, 64)
    assert torch.isfinite(out).all()


def test_rotational_slicegpt_int4_reduction():
    model = DummyMLP()
    compressor = RotationalSliceGPTINT4Compressor(
        model,
        rank=16,
        quant_bits=4,
        max_error=2.0,
        fallback_to_original=False,
    )
    compressed, stats = compressor.compress()

    # Should have achieved physical reduction or valid int4 bit quantization
    has_quant = any(getattr(m, "effective_bits", 32) <= 8 for m in compressed.modules())
    orig_params = sum(p.numel() for p in model.parameters())
    comp_params = sum(p.numel() for p in compressed.parameters())
    effective_ratio = (orig_params / max(comp_params, 1)) * (32.0 / 4.0 if has_quant else 1.0)
    assert effective_ratio > 1.05

    x = torch.randn(2, 128)
    out = compressed(x)
    assert out.shape == (2, 64)
    assert torch.isfinite(out).all()


def test_rotational_slicegpt_int4_factory():
    model = DummyMLP()
    comp_model, stats = rotational_slicegpt_int4_factory(
        model,
        rank=16,
        quant_bits=4,
        max_error=2.0,
        fallback_to_original=False,
        return_stats=True,
    )
    assert comp_model is not None
    assert isinstance(stats, dict)
    x = torch.randn(2, 128)
    out = comp_model(x)
    assert out.shape == (2, 64)
    assert torch.isfinite(out).all()
