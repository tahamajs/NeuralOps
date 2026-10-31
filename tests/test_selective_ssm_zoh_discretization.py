#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for Selective SSM ZOH Discretization and Recurrent State Stability
"""

import math
import pytest
import torch
from neuralops.layers.ssm.selective_ssm import (
    SelectiveSSM,
    stable_zoh_b,
    discretize_zoh_b,
)
from neuralops.core.selective_scan import selective_scan_chunked


def test_stable_zoh_b_numerical_accuracy():
    """Verify Taylor series expansion matches exact formulation across scale regimes."""
    # Test dA = 0: limit is 1.0
    zero_da = torch.tensor([0.0], dtype=torch.float32)
    val_zero = stable_zoh_b(zero_da)
    assert torch.isclose(val_zero, torch.tensor([1.0])), f"Expected 1.0, got {val_zero}"

    # Test small dA regime (|dA| < 1e-4) where Taylor expansion avoids catastrophic cancellation
    small_da = torch.tensor([1e-6, -1e-6, 5e-5, -5e-5], dtype=torch.float64)
    phi_small = stable_zoh_b(small_da)
    # Double precision exact ground truth: (exp(x) - 1) / x
    expected_small = torch.expm1(small_da) / small_da
    assert torch.allclose(phi_small, expected_small, atol=1e-12)

    # Test general dA regime (|dA| >= 1e-4)
    general_da = torch.tensor([-0.01, -0.1, -1.0, -5.0], dtype=torch.float32)
    phi_general = stable_zoh_b(general_da)
    expected_general = torch.expm1(general_da) / general_da
    assert torch.allclose(phi_general, expected_general, atol=1e-6)

    # Test alias identity
    assert discretize_zoh_b is stable_zoh_b


def test_selective_ssm_initialization():
    """Verify SelectiveSSM block registers and initializes with valid parameters."""
    block = SelectiveSSM(d_model=64, d_state=16, d_conv=4, expand=2)
    assert block.d_inner == 128
    assert block.A_log.shape == (128, 16)
    assert block.D.shape == (128,)


def test_selective_ssm_forward_shape_and_finite():
    """Verify SelectiveSSM forward pass produces correctly shaped finite tensors."""
    block = SelectiveSSM(d_model=64, d_state=16, d_conv=4, expand=2)
    x = torch.randn(2, 32, 64)
    out, present = block(x)
    assert out.shape == (2, 32, 64)
    assert torch.isfinite(out).all()


def test_selective_ssm_long_sequence_no_drift():
    """Verify recurrent state accumulator remains bounded and does not drift for N=4096."""
    torch.manual_seed(42)
    d_inner = 16
    d_state = 8
    seq_len = 4096

    u = torch.randn(1, d_inner, seq_len) * 0.1
    # Small delta values that would trigger Euler / naive discretization drift
    delta = torch.full((1, d_inner, seq_len), 0.005)
    # Negative A eigenvalues for continuous stable state matrix
    A = -torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
    B = torch.randn(1, d_state, seq_len) * 0.1
    C = torch.randn(1, d_state, seq_len) * 0.1

    y = selective_scan_chunked(u, delta, A, B, C, delta_softplus=False)
    assert y.shape == (1, d_inner, seq_len)
    assert torch.isfinite(y).all()
    # Ensure recurrent state has not blown up to infinity
    max_abs = y.abs().max().item()
    assert max_abs < 100.0, f"Recurrent state output drifted: max_abs={max_abs}"
