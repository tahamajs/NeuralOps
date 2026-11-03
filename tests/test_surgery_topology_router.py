#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Unit tests for Phase 3 TopologyRouter, padding sanitation, and bidirectional wrapping."""

import torch
import torch.nn as nn
import pytest

from neuralops.surgery.topology_router import (
    extract_binary_token_mask,
    sanitize_kv_for_padding_mask,
    BiDirectionalScanWrapper,
    AsymmetricCrossAttentionRouter,
    TopologyRouter,
)
from neuralops.attentions.linear.cosformer import CosformerAttention


def test_extract_binary_token_mask_2d_and_4d():
    mask_2d = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.int64)
    keep_2d = extract_binary_token_mask(mask_2d, seq_len=4)
    assert keep_2d.shape == (2, 1, 4, 1)
    assert keep_2d[0, 0, 3, 0] == 0.0
    assert keep_2d[0, 0, 0, 0] == 1.0

    mask_4d = torch.tensor([[[[0.0, 0.0, 0.0, -10000.0]]]])
    keep_4d = extract_binary_token_mask(mask_4d, seq_len=4)
    assert keep_4d.shape == (1, 1, 4, 1)
    assert keep_4d[0, 0, 3, 0] == 0.0
    assert keep_4d[0, 0, 0, 0] == 1.0


def test_padding_sanitization_zeroing():
    B, H, T, D = 2, 2, 4, 16
    k = torch.ones(B, H, T, D)
    v = torch.ones(B, H, T, D)
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.int64)

    k_clean, v_clean = sanitize_kv_for_padding_mask(k, v, mask)
    assert torch.all(k_clean[0, :, 3, :] == 0.0)
    assert torch.all(v_clean[0, :, 3, :] == 0.0)
    assert torch.all(k_clean[0, :, 0, :] == 1.0)
    assert torch.all(v_clean[0, :, 0, :] == 1.0)


def test_bidirectional_scan_wrapper():
    cosformer = CosformerAttention(hidden_size=64, num_heads=4)
    bi_wrapper = BiDirectionalScanWrapper(cosformer, d_model=64)

    B, T, D = 2, 8, 64
    x = torch.randn(B, T, D, requires_grad=True)
    mask = torch.ones(B, T, dtype=torch.int64)
    mask[0, 6:] = 0

    out = bi_wrapper(x, attention_mask=mask)
    if isinstance(out, tuple):
        out = out[0]
    assert out.shape == (B, T, D)

    loss = out.sum()
    loss.backward()
    assert x.grad is not None


def test_asymmetric_cross_attention_router():
    router = AsymmetricCrossAttentionRouter(d_model=64, num_heads=4)
    B, T_q, T_kv, D = 2, 16, 32, 64
    q_hidden = torch.randn(B, T_q, D)
    kv_hidden = torch.randn(B, T_kv, D)

    out, _ = router(q_hidden, encoder_hidden_states=kv_hidden)
    assert out.shape == (B, T_q, D)


def test_topology_router_alternating_execution():
    """Phase 055: Tests TopologyRouter alternating routing without shape mismatches."""
    linear_branch = nn.Identity()
    full_attn_branch = nn.Linear(64, 64)
    router = TopologyRouter(
        branches={"linear": linear_branch, "full": full_attn_branch},
        routing_strategy="alternating",
    )

    B, T, D = 2, 8, 64
    x = torch.randn(B, T, D)

    # Even layer -> 'linear'
    out_even = router(x, layer_idx=0)
    assert out_even.shape == (B, T, D)
    assert torch.allclose(out_even, x)

    # Odd layer -> 'full'
    out_odd = router(x, layer_idx=1)
    assert out_odd.shape == (B, T, D)


def test_topology_router_layer_schedule_and_sliding_window():
    """Phase 055: Tests TopologyRouter layer schedule mapping and sliding window thresholding."""
    branch_a = nn.Identity()
    branch_b = nn.Linear(32, 32)
    schedule = {0: "window", 1: "linear", 2: "window"}
    router = TopologyRouter(
        branches={"window": branch_a, "linear": branch_b},
        routing_strategy="layer_schedule",
        schedule=schedule,
        window_size=16,
    )

    B, T, D = 2, 10, 32
    x = torch.randn(B, T, D)

    # Scheduled layer 0 -> window
    out_0 = router(x, layer_idx=0)
    assert out_0.shape == (B, T, D)
    assert torch.allclose(out_0, x)

    # Scheduled layer 1 -> linear
    out_1 = router(x, layer_idx=1)
    assert out_1.shape == (B, T, D)

    # Switch to sliding window strategy
    router.routing_strategy = "sliding_window"
    # T=10 <= window_size=16 -> window (identity)
    out_short = router(x)
    assert torch.allclose(out_short, x)

    # T=32 > window_size=16 -> linear
    x_long = torch.randn(B, 32, D)
    out_long = router(x_long)
    assert out_long.shape == (B, 32, D)


def test_topology_router_dynamic_gate_moa():
    """Phase 055: Tests Dynamic Mixture-of-Attentions gating."""
    branch_1 = nn.Linear(32, 32)
    branch_2 = nn.Linear(32, 32)
    router = TopologyRouter(
        branches={"branch_1": branch_1, "branch_2": branch_2},
        routing_strategy="dynamic_gate",
        d_model=32,
    )

    B, T, D = 2, 8, 32
    x = torch.randn(B, T, D)
    out = router(x)
    assert out.shape == (B, T, D)


def test_topology_router_shape_mismatch_exception():
    """Phase 055: Guarantees TopologyRouter raises ValueError upon shape mismatch."""
    invalid_branch = nn.Linear(32, 16)  # outputs D=16 instead of 32
    router = TopologyRouter(
        branches={"bad_branch": invalid_branch},
        routing_strategy="passthrough",
    )

    B, T, D = 2, 8, 32
    x = torch.randn(B, T, D)
    with pytest.raises(ValueError, match="output shape mismatch"):
        router(x)

