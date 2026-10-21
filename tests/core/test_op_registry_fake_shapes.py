#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Shape-contract tests for FakeTensor implementations in the core op registry."""

from __future__ import annotations

import pytest
import torch

from neuralops.core.triton_kernels import op_registry


@pytest.mark.parametrize(
    ("fake_name", "extra_args"),
    [
        ("_op_sparse_attn_fake", ()),
        ("_op_sliding_window_attn_fake", ()),
        ("_op_block_sparse_attn_fake", (torch.empty((2, 3), device="meta"),)),
        (
            "_op_varlen_attn_fake",
            (
                torch.empty((3,), dtype=torch.int32, device="meta"),
                torch.empty((3,), dtype=torch.int32, device="meta"),
            ),
        ),
    ],
)
def test_attention_fake_shapes_follow_value_dimension(fake_name: str, extra_args: tuple) -> None:
    q = torch.empty((2, 4, 7, 8), device="meta")
    k = torch.empty((2, 4, 7, 8), device="meta")
    v = torch.empty((2, 4, 7, 5), device="meta")
    fake = getattr(op_registry, fake_name)

    result = fake(q, k, v, *extra_args)

    assert result.shape == (2, 4, 7, 5)


def test_gqa_fake_rejects_non_divisible_head_count() -> None:
    k = torch.empty((1, 3, 5, 8), device="meta")
    v = torch.empty((1, 3, 5, 6), device="meta")

    with pytest.raises(ValueError, match="divisible"):
        op_registry._op_gqa_fake(k, v, num_kv_heads=3, num_heads=8)


def test_gqa_fake_repeats_heads_to_requested_count() -> None:
    k = torch.empty((1, 2, 5, 8), device="meta")
    v = torch.empty((1, 2, 5, 6), device="meta")

    k_out, v_out = op_registry._op_gqa_fake(k, v, num_kv_heads=2, num_heads=8)

    assert k_out.shape == (1, 8, 5, 8)
    assert v_out.shape == (1, 8, 5, 6)
