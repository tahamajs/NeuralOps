#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for Example 12: Structured Pruning, 2:4 Hardware Sparsity, and Token Shedding.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
import torch
import torch.nn as nn

from examples._shared import TinyClassifier

ex12 = importlib.import_module("examples.12_structured_pruning_and_token_shedding")
apply_24_sparsity = ex12.apply_24_sparsity
prune_mlp_intermediate = ex12.prune_mlp_intermediate
TokenSheddingWrapper = ex12.TokenSheddingWrapper


def test_12_runs_clean():
    """Verify that example 12 executes cleanly as a subprocess and exits 0."""
    proc = subprocess.run(
        [sys.executable, "examples/12_structured_pruning_and_token_shedding.py"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, f"Example 12 failed with stderr: {proc.stderr}"
    assert "[PASS] Overall Example 12 Status" in proc.stdout


def test_12_sparsity_2_4_exact():
    """Verify strict 2:4 sparsity pattern and exactly 50% zero elements."""
    model = TinyClassifier().to("cpu")
    weight = model.transformer.distilbert.transformer.layer[0].attention.q_lin.weight.detach().clone()
    m, n = weight.shape
    assert n % 4 == 0

    w_sparse, mask_24, sparsity = apply_24_sparsity(weight)

    # Invariant: for every 4 contiguous elements along each row, exactly 2 are zero
    zeros_per_block = (w_sparse.view(m, n // 4, 4) == 0).sum(dim=-1)
    assert (zeros_per_block == 2).all().item(), "2:4 sparsity pattern violated"
    assert abs(sparsity - 0.50) < 1e-6, f"Expected 0.50 sparsity ratio, got {sparsity}"
    assert (w_sparse == 0).sum().item() == weight.numel() // 2


def test_12_mlp_channel_pruning():
    """Verify structured channel pruning on linear layers with parameter reduction."""
    lin1 = nn.Linear(32, 64, bias=False)
    lin2 = nn.Linear(64, 32, bias=False)
    new_lin1, new_lin2, p_old, p_new = prune_mlp_intermediate(lin1, lin2, prune_ratio=0.5)

    assert new_lin1.out_features == 32
    assert new_lin2.in_features == 32
    assert p_new < p_old
    assert p_new == (32 * 32) + (32 * 32)

    x = torch.ones(2, 32)
    out = new_lin2(torch.relu(new_lin1(x)))
    assert out.shape == (2, 32)
    assert torch.isfinite(out).all()


def test_12_token_shedding_preserves_sink():
    """Verify token shedding retains sink token (index 0) and reduces sequence length."""
    bsz, seq_len, dim = 2, 16, 32
    hidden_states = torch.ones(bsz, seq_len, dim)
    # Give token 0 zero attention to test that it is retained via sink protection
    attn_weights = torch.ones(bsz, 2, seq_len, seq_len)
    attn_weights[:, :, :, 0] = 0.0

    shedder = TokenSheddingWrapper(keep_ratio=0.5)
    shortened, retained_idx, red_pct = shedder(hidden_states, attn_weights)

    assert shortened.shape == (bsz, 8, dim)
    assert (retained_idx[:, 0] == 0).all().item(), "Token 0 sink token must always be retained"
    assert red_pct == 50.0
