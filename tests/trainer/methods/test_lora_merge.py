#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for LoRA weight merging and unmerging bitwise roundtrip."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from neuralops.trainer.methods.lora import LoRALinear


def test_lora_merge_unmerge_bitwise_wrapper():
    """Verify merge() then unmerge() gives back original weights bitwise for wrapper mode."""
    torch.manual_seed(42)
    base = nn.Linear(64, 32, bias=True)
    orig_weight = base.weight.clone()
    orig_bias = base.bias.clone()

    lora = LoRALinear(base, r=8, alpha=16.0, dropout=0.0)

    # Perturb lora weights so delta is non-zero
    with torch.no_grad():
        lora.lora_A.data.normal_(0.0, 0.1)
        lora.lora_B.data.normal_(0.0, 0.1)

    # Merge
    lora.merge()
    assert lora.merged is True
    # Base weight must have changed
    assert not torch.equal(base.weight, orig_weight)

    # Unmerge
    lora.unmerge()
    assert lora.merged is False
    # Base weight must be bitwise identical to orig_weight
    assert torch.equal(base.weight, orig_weight)
    assert torch.equal(base.bias, orig_bias)


def test_lora_merge_unmerge_bitwise_standalone():
    """Verify merge() then unmerge() gives back original weights bitwise for standalone mode."""
    torch.manual_seed(42)
    lora = LoRALinear(in_features=64, out_features=32, lora_rank=8, lora_alpha=16.0, bias=True)
    orig_weight = lora.weight.clone()
    orig_bias = lora.bias.clone()

    with torch.no_grad():
        lora.lora_A.data.normal_(0.0, 0.1)
        lora.lora_B.data.normal_(0.0, 0.1)

    lora.merge()
    assert lora.merged is True
    assert not torch.equal(lora.weight, orig_weight)

    lora.unmerge()
    assert lora.merged is False
    assert torch.equal(lora.weight, orig_weight)
    assert torch.equal(lora.bias, orig_bias)


def test_lora_merge_math_formula():
    """Verify base.weight += (lora_B @ lora_A) * scaling exactly on merge."""
    torch.manual_seed(42)
    base = nn.Linear(32, 16, bias=False)
    orig_weight = base.weight.clone()

    r = 4
    alpha = 8.0
    scaling = alpha / r
    lora = LoRALinear(base, r=r, alpha=alpha, dropout=0.0)

    with torch.no_grad():
        lora.lora_A.data.normal_(0.0, 0.2)
        lora.lora_B.data.normal_(0.0, 0.2)
        expected_delta = (lora.lora_B @ lora.lora_A) * scaling
        expected_merged = orig_weight + expected_delta

    lora.merge()
    assert torch.allclose(base.weight, expected_merged, atol=1e-6)


def test_lora_merge_idempotence():
    """Verify calling merge() multiple times only merges once."""
    torch.manual_seed(42)
    base = nn.Linear(32, 16)
    lora = LoRALinear(base, r=4, alpha=8.0)
    with torch.no_grad():
        lora.lora_B.data.normal_(0.0, 0.2)

    lora.merge()
    merged_once = base.weight.clone()
    lora.merge()
    assert torch.equal(base.weight, merged_once)


def test_lora_unmerge_idempotence():
    """Verify calling unmerge() multiple times only unmerges once."""
    torch.manual_seed(42)
    base = nn.Linear(32, 16)
    orig_weight = base.weight.clone()
    lora = LoRALinear(base, r=4, alpha=8.0)
    with torch.no_grad():
        lora.lora_B.data.normal_(0.0, 0.2)

    lora.merge()
    lora.unmerge()
    assert torch.equal(base.weight, orig_weight)
    lora.unmerge()
    assert torch.equal(base.weight, orig_weight)


def test_lora_merge_forward_equivalence():
    """Verify forward(x) after merge produces identical output to forward(x) before merge."""
    torch.manual_seed(42)
    base = nn.Linear(32, 16)
    lora = LoRALinear(base, r=4, alpha=8.0, dropout=0.0)
    with torch.no_grad():
        lora.lora_A.data.normal_(0.0, 0.1)
        lora.lora_B.data.normal_(0.0, 0.1)

    x = torch.randn(4, 32)
    unmerged_out = lora(x)

    lora.merge()
    merged_out = lora(x)

    assert torch.allclose(unmerged_out, merged_out, atol=1e-5)
