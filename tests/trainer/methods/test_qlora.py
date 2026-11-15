#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for Quantized Linear and QLoRA fine-tuning method."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.trainer.methods.qlora import QLoRALinear, QuantizedLinear, apply_qlora


def test_quantized_linear_4bit_and_8bit():
    """Verify QuantizedLinear quantization and dequantization in 4-bit and 8-bit."""
    torch.manual_seed(42)
    base = nn.Linear(64, 32, bias=True)

    # 4-bit
    qlin_4bit = QuantizedLinear.from_linear(base, bits=4, block_size=16)
    x = torch.randn(4, 64)
    base_out = base(x)
    q4_out = qlin_4bit(x)
    sim_4bit = F.cosine_similarity(base_out.flatten(), q4_out.flatten(), dim=0)
    assert sim_4bit.item() > 0.95

    # 8-bit
    qlin_8bit = QuantizedLinear.from_linear(base, bits=8, block_size=16)
    q8_out = qlin_8bit(x)
    sim_8bit = F.cosine_similarity(base_out.flatten(), q8_out.flatten(), dim=0)
    assert sim_8bit.item() > 0.99


def test_qlora_linear_r_and_rank_attributes():
    """Verify QLoRALinear accepts both rank and r arguments and sets both attributes."""
    base = nn.Linear(32, 16)
    qlora1 = QLoRALinear.from_linear(base, r=8, alpha=16.0)
    assert qlora1.r == 8
    assert qlora1.rank == 8
    assert qlora1.scaling == 2.0

    qlora2 = QLoRALinear(32, 16, r=4, alpha=8.0)
    assert qlora2.r == 4
    assert qlora2.rank == 4
    assert qlora2.scaling == 2.0


def test_qlora_forward_with_adapter_delta():
    """Verify forward() adds adapter output to quantized base output."""
    torch.manual_seed(42)
    base = nn.Linear(32, 16)
    qlora = QLoRALinear.from_linear(base, rank=4, alpha=8.0)

    # At init, lora_B is zeros, so forward matches base exactly
    x = torch.randn(2, 32)
    base_q_out = qlora.base(x)
    assert torch.equal(qlora(x), base_q_out)

    # Non-zero adapter weights
    with torch.no_grad():
        qlora.lora_A.weight.normal_(0.0, 0.1)
        qlora.lora_B.weight.normal_(0.0, 0.1)

    expected_adapter = qlora.lora_B(qlora.lora_A(x)) * qlora.scaling
    expected_out = qlora.base(x) + expected_adapter
    assert torch.allclose(qlora(x), expected_out, atol=1e-5)


def test_qlora_gradient_isolation():
    """Verify base buffers are non-trainable and only adapters receive gradients."""
    base = nn.Linear(32, 16)
    qlora = QLoRALinear.from_linear(base, rank=4, alpha=8.0)

    x = torch.randn(2, 32)
    out = qlora(x)
    loss = out.sum()
    loss.backward()

    assert not qlora.base.weight_q.requires_grad
    assert not qlora.base.scales.requires_grad
    assert qlora.lora_A.weight.grad is not None
    assert qlora.lora_B.weight.grad is not None
