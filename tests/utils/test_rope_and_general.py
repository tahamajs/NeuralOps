#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Unit tests for neuralops.utils.rope and neuralops.utils.general."""

from __future__ import annotations

import torch
import torch.nn as nn

from neuralops.utils.rope import (
    rotate_half,
    apply_rotary_pos_emb,
    broadcast_cos_sin,
)
from neuralops.utils.general import (
    get_device,
    set_seed,
    count_parameters,
    format_number,
    ste_round,
    ste_clamp,
)


def test_rotate_half_even_and_odd():
    """Verify half rotation on even and odd head dimensions."""
    # Even dimension
    x_even = torch.tensor([1.0, 2.0, 3.0, 4.0])
    rot_even = rotate_half(x_even)
    # x1 = [1, 2], x2 = [3, 4] -> [-x2, x1] = [-3, -4, 1, 2]
    assert torch.equal(rot_even, torch.tensor([-3.0, -4.0, 1.0, 2.0]))

    # Odd dimension (half = 2, tail = [5.0])
    x_odd = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    rot_odd = rotate_half(x_odd)
    assert torch.equal(rot_odd, torch.tensor([-3.0, -4.0, 1.0, 2.0, 5.0]))


def test_apply_rotary_pos_emb():
    """Verify RoPE application computes (x * cos) + (rotate_half(x) * sin)."""
    b, s, h, d = 2, 4, 8, 16
    q = torch.randn(b, s, h, d)
    k = torch.randn(b, s, h, d)
    cos = torch.ones(b, s, 1, d)
    sin = torch.zeros(b, s, 1, d)

    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
    # With cos=1 and sin=0, rotary embeddings should leave tensors unchanged
    assert torch.allclose(q_rot, q)
    assert torch.allclose(k_rot, k)


def test_broadcast_cos_sin():
    """Verify cos and sin broadcasting to target query shape."""
    b, s, h, d = 2, 8, 4, 32
    q = torch.randn(b, s, h, d)
    cos = torch.ones(s, d)
    sin = torch.zeros(s, d)

    b_cos, b_sin = broadcast_cos_sin(cos, sin, q)
    assert b_cos.shape == (1, 1, s, d)
    assert b_sin.shape == (1, 1, s, d)


def test_general_utils():
    """Verify device resolution, parameter counting, formatting, and STE."""
    # get_device
    cpu_dev = get_device("cpu")
    assert cpu_dev == torch.device("cpu")

    # set_seed
    set_seed(42)

    # count_parameters
    lin = nn.Linear(10, 20)  # 10*20 + 20 = 220 params
    assert count_parameters(lin) == 220

    # format_number
    assert format_number(1_500_000) == "1.5M"
    assert format_number(3_000) == "3K"

    # ste_round and ste_clamp
    t = torch.tensor([1.2, 2.7, 3.5], requires_grad=True)
    rounded = ste_round(t)
    assert torch.equal(rounded, torch.tensor([1.0, 3.0, 4.0]))
    rounded.sum().backward()
    assert torch.equal(t.grad, torch.ones_like(t))
