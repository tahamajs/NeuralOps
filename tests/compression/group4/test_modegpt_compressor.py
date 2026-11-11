#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit tests for MoDeGPTCompressor (Group 4, Key 54: modegpt_compressor).
"""

import pytest
import torch
import torch.nn as nn
from neuralops.compression.low_rank.modegpt_compressor import (
    MoDeGPTCompressor,
    modegpt_factory,
    MoDeGPTLinear,
)
from neuralops.core.registry import COMPRESSOR_REGISTRY


class DummyMLP(nn.Module):
    def __init__(self, in_features: int = 128, out_features: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(out_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


def test_registry():
    assert "modegpt_compressor" in COMPRESSOR_REGISTRY
    assert COMPRESSOR_REGISTRY.get("modegpt_compressor") is not None


def test_modegpt_linear_forward():
    lin = nn.Linear(128, 128)
    layer = MoDeGPTLinear.from_linear(lin, rank=16)
    x = torch.randn(2, 128)
    out = layer(x)
    assert out.shape == (2, 128)
    assert torch.isfinite(out).all()


def test_modegpt_compressor():
    model = DummyMLP()
    comp = MoDeGPTCompressor(model, rank=16)
    compressed, stats = comp.compress()
    assert compressed is not None
    assert stats["forward_check_success"] is True

    x = torch.randn(2, 128)
    out = compressed(x)
    assert out.shape == (2, 128)
    assert torch.isfinite(out).all()


def test_modegpt_factory():
    model = DummyMLP()
    compressed = modegpt_factory(model, rank=16)
    assert compressed is not None
    x = torch.randn(2, 128)
    out = compressed(x)
    assert torch.isfinite(out).all()


def test_modegpt_lora_bias_single_count_and_mixed_dtype():
    torch.manual_seed(0)
    lin = nn.Linear(64, 48, bias=True)
    layer = MoDeGPTLinear.from_linear(lin, rank=16, lora_r=4, lora_alpha=8.0)
    with torch.no_grad():
        layer.lora_B.normal_(std=0.1)

    x = torch.randn(3, 64, requires_grad=True)
    out = layer(x)

    v = layer._V()
    expected = (
        torch.nn.functional.linear(torch.nn.functional.linear(x, v), layer.U, layer.bias)
        + torch.nn.functional.linear(torch.nn.functional.linear(x, layer.lora_A), layer.lora_B) * layer.scaling
    )
    assert torch.allclose(out, expected, atol=1e-5)

    out.pow(2).mean().backward()
    assert layer.lora_A.grad is not None and layer.lora_A.grad.abs().sum().item() > 0
    assert layer.lora_B.grad is not None and layer.lora_B.grad.abs().sum().item() > 0

    # Mixed dtype input (bfloat16) against float32 layer weights must not raise
    x_bf16 = torch.randn(2, 64, dtype=torch.bfloat16)
    out_bf16 = layer(x_bf16)
    assert out_bf16.dtype == torch.bfloat16
    assert torch.isfinite(out_bf16).all()


def test_modegpt_shared_pair_compressor_with_lora():
    class PairedBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate_proj = nn.Linear(64, 96, bias=False)
            self.up_proj = nn.Linear(64, 96, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.gate_proj(x) * self.up_proj(x)

    torch.manual_seed(7)
    blk = PairedBlock()
    comp = MoDeGPTCompressor(blk, rank=16, lora_r=4, max_error=0.95)
    compressed, stats = comp.compress()
    assert stats["forward_check_success"] is True
    assert compressed.gate_proj._uses_shared_V is True
    assert compressed.up_proj._uses_shared_V is True

    x = torch.randn(2, 8, 64, requires_grad=True)
    y = compressed(x)
    assert y.shape == (2, 8, 96)
    assert torch.isfinite(y).all()
    y.sum().backward()
    assert compressed.gate_proj.lora_B.grad is not None
    assert compressed.up_proj.lora_B.grad is not None
