#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for NeuralOpsRMSNorm dtype matching."""

from __future__ import annotations

import pytest
import torch

from neuralops.models.modeling_norm import NeuralOpsRMSNorm


@pytest.mark.parametrize("input_dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_rmsnorm_preserves_input_dtype_with_fp32_weights(input_dtype: torch.dtype) -> None:
    norm = NeuralOpsRMSNorm(hidden_size=64)
    # Ensure parameter is in float32 (standard initialization)
    assert norm.weight.dtype == torch.float32

    x = torch.randn(2, 8, 64, dtype=input_dtype)
    out = norm(x)

    assert out.dtype == input_dtype
    assert out.shape == x.shape
    assert not torch.isnan(out).any()


def test_rmsnorm_with_cast_weights() -> None:
    norm = NeuralOpsRMSNorm(hidden_size=32)
    norm.weight.data = norm.weight.data.to(torch.float16)

    # Input is float32 while weight is float16
    x = torch.randn(1, 4, 32, dtype=torch.float32)
    out = norm(x)

    assert out.dtype == torch.float32
    assert out.shape == x.shape
