#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for LoRA forward pass numerical calculation and configuration."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.trainer.methods.lora import LoRALinear


def test_lora_forward_adds_scaled_adapter():
    """Verify forward() adds (lora_B @ lora_A) @ x to base output, scaled by alpha / r."""
    torch.manual_seed(42)
    in_features, out_features, r, alpha = 32, 16, 4, 8.0
    scaling = alpha / r

    base = nn.Linear(in_features, out_features, bias=True)
    lora = LoRALinear(base, r=r, alpha=alpha, dropout=0.0)

    # Give non-zero weights to adapters
    with torch.no_grad():
        lora.lora_A.data.normal_(0.0, 0.2)
        lora.lora_B.data.normal_(0.0, 0.2)

    x = torch.randn(8, in_features)
    out = lora(x)

    # Manual calculation: base(x) + ((x @ A.T) @ B.T) * scaling
    base_out = base(x)
    delta_w = (lora.lora_B @ lora.lora_A) * scaling
    expected_out = base_out + F.linear(x, delta_w)

    assert torch.allclose(out, expected_out, atol=1e-5)


def test_lora_r_and_alpha_parameters_respected():
    """Verify r and alpha scaling factor is strictly respected."""
    torch.manual_seed(42)
    base = nn.Linear(32, 16, bias=False)

    lora1 = LoRALinear(base, r=4, alpha=4.0, dropout=0.0)
    lora2 = LoRALinear(base, r=4, alpha=8.0, dropout=0.0)

    assert lora1.scaling == 1.0
    assert lora2.scaling == 2.0
    assert lora1.r == 4
    assert lora2.r == 4
    assert lora1.alpha == 4.0
    assert lora2.alpha == 8.0

    with torch.no_grad():
        w_A = torch.randn(4, 32)
        w_B = torch.randn(16, 4)
        lora1.lora_A.data.copy_(w_A)
        lora1.lora_B.data.copy_(w_B)
        lora2.lora_A.data.copy_(w_A)
        lora2.lora_B.data.copy_(w_B)

    x = torch.randn(2, 32)
    base_out = base(x)
    diff1 = lora1(x) - base_out
    diff2 = lora2(x) - base_out

    # lora2 has double the alpha, so diff2 must be exactly double diff1
    assert torch.allclose(diff2, diff1 * 2.0, atol=1e-5)


def test_lora_wrapper_positional_arguments():
    """Verify LoRALinear handles positional arguments LoRALinear(module, r, alpha, dropout)."""
    base = nn.Linear(32, 16)
    lora = LoRALinear(base, 4, 8.0, 0.05)

    assert lora.r == 4
    assert lora.alpha == 8.0
    assert lora.scaling == 2.0
    assert lora.in_features == 32
    assert lora.out_features == 16


def test_lora_zero_initialization():
    """Verify LoRA initialization produces zero delta and matches base exactly."""
    base = nn.Linear(32, 16)
    lora = LoRALinear(base, r=4, alpha=8.0, dropout=0.0)

    x = torch.randn(4, 32)
    assert torch.equal(lora(x), base(x))


def test_lora_gradient_flow_and_freezing():
    """Verify base weights stay frozen while adapters receive gradients."""
    base = nn.Linear(32, 16)
    lora = LoRALinear(base, r=4, alpha=8.0)

    x = torch.randn(2, 32)
    out = lora(x)
    out.sum().backward()

    assert base.weight.grad is None
    assert lora.lora_A.weight.grad is not None
    assert lora.lora_B.weight.grad is not None
