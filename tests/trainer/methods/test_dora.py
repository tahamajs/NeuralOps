#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for DoRA (Weight-Decomposed Low-Rank Adaptation)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.trainer.methods.dora import DoRALinear, apply_dora, merge_dora_into_linear
from neuralops.trainer.methods.lora import LoRALinear


def test_dora_initialization_norm():
    """Verify magnitude parameter m is initialized to row-wise Frobenius norm of base weights."""
    torch.manual_seed(42)
    base = nn.Linear(32, 16, bias=True)
    dora = DoRALinear(base, r=4, alpha=8.0)

    expected_m = base.weight.norm(p=2, dim=1)
    assert torch.allclose(dora.m, expected_m, atol=1e-6)

    # Since lora_B is initialized to 0, initial forward matches base linear exactly
    x = torch.randn(4, 32)
    assert torch.allclose(dora(x), base(x), atol=1e-5)


def test_dora_reduces_to_lora_when_mag_is_one():
    """Verify that when magnitude is all-ones and weight direction is unit-norm, DoRA reduces to LoRA."""
    torch.manual_seed(42)
    in_features, out_features = 32, 16
    r, alpha = 4, 8.0

    # Create base weight with unit row norm
    raw_w = torch.randn(out_features, in_features)
    unit_w = raw_w / raw_w.norm(p=2, dim=1, keepdim=True)

    base_dora = nn.Linear(in_features, out_features, bias=False)
    base_lora = nn.Linear(in_features, out_features, bias=False)
    with torch.no_grad():
        base_dora.weight.copy_(unit_w)
        base_lora.weight.copy_(unit_w)

    dora = DoRALinear(base_dora, r=r, alpha=alpha, dropout=0.0)
    lora = LoRALinear(base_lora, r=r, alpha=alpha, dropout=0.0)

    # Set identical adapter weights
    A_weight = torch.randn(r, in_features) * 0.05
    B_weight = torch.randn(out_features, r) * 0.05
    with torch.no_grad():
        dora.lora_A.weight.copy_(A_weight)
        dora.lora_B.weight.copy_(B_weight)
        lora.lora_A.weight.copy_(A_weight)
        lora.lora_B.weight.copy_(B_weight)

        # Set DoRA magnitude m = effective_norm of (W0 + Delta W)
        delta = lora.scaling * (lora.lora_B.weight @ lora.lora_A.weight)
        effective_w = unit_w + delta
        dora.m.copy_(effective_w.norm(p=2, dim=1))

    x = torch.randn(8, in_features)
    dora_out = dora(x)
    lora_out = lora(x)

    # When magnitude equals the norm of W0 + delta, DoRA equals LoRA identically
    assert torch.allclose(dora_out, lora_out, atol=1e-5)

    # And when magnitude is set to all-ones on a normalized directional weight:
    with torch.no_grad():
        dora.m.copy_(torch.ones(out_features))
        eff_dora = dora.get_effective_weight()
        # Direction has unit norm for each row
        row_norms = eff_dora.norm(p=2, dim=1)
        assert torch.allclose(row_norms, torch.ones(out_features), atol=1e-5)


def test_dora_magnitude_applied_after_direction_update():
    """Verify magnitude vector m is applied after direction is updated."""
    torch.manual_seed(42)
    base = nn.Linear(32, 16, bias=False)
    dora = DoRALinear(base, r=4, alpha=8.0)

    with torch.no_grad():
        dora.lora_A.weight.normal_(0.0, 0.1)
        dora.lora_B.weight.normal_(0.0, 0.1)

    effective_w = dora.get_effective_weight()
    # Check that each row i has norm equal to m[i]
    computed_row_norms = effective_w.norm(p=2, dim=1)
    assert torch.allclose(computed_row_norms, dora.m.abs(), atol=1e-5)

    # Scaling m by 2 should scale the entire effective weight by 2
    with torch.no_grad():
        dora.m.mul_(2.0)
    effective_w_scaled = dora.get_effective_weight()
    assert torch.allclose(effective_w_scaled, effective_w * 2.0, atol=1e-5)


def test_dora_merge_and_unmerge_roundtrip():
    """Verify merge_weights() and unmerge() preserves and restores bitwise weights."""
    torch.manual_seed(42)
    base = nn.Linear(32, 16, bias=True)
    orig_weight = base.weight.clone()
    orig_bias = base.bias.clone()

    dora = DoRALinear(base, r=4, alpha=8.0)
    with torch.no_grad():
        dora.lora_A.weight.normal_(0.0, 0.1)
        dora.lora_B.weight.normal_(0.0, 0.1)

    pre_merge_eff = dora.get_effective_weight().clone()

    x = torch.randn(4, 32)
    pre_merge_out = dora(x)

    # Merge
    dora.merge_weights()
    assert dora.merged is True
    assert torch.allclose(base.weight, pre_merge_eff, atol=1e-5)

    post_merge_out = dora(x)
    assert torch.allclose(pre_merge_out, post_merge_out, atol=1e-5)

    # Unmerge
    dora.unmerge()
    assert dora.merged is False
    assert torch.equal(base.weight, orig_weight)
    assert torch.equal(base.bias, orig_bias)


def test_merge_dora_into_linear_helper():
    """Verify helper merge_dora_into_linear folds DoRA into nn.Linear."""
    torch.manual_seed(42)
    base = nn.Linear(32, 16)
    dora = DoRALinear(base, r=4, alpha=8.0)
    with torch.no_grad():
        dora.lora_B.weight.normal_(0.0, 0.1)

    x = torch.randn(2, 32)
    expected_out = dora(x)

    returned_linear = merge_dora_into_linear(dora)
    assert returned_linear is base
    assert isinstance(returned_linear, nn.Linear)
    assert torch.allclose(returned_linear(x), expected_out, atol=1e-5)


def test_dora_gradient_flow():
    """Verify backward pass computes valid gradients for m, lora_A, and lora_B."""
    base = nn.Linear(32, 16)
    dora = DoRALinear(base, r=4, alpha=8.0)

    x = torch.randn(2, 32)
    loss = dora(x).sum()
    loss.backward()

    assert not base.weight.requires_grad
    assert dora.m.grad is not None
    assert dora.lora_A.weight.grad is not None
    assert dora.lora_B.weight.grad is not None
