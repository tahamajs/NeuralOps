#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for elementwise kernels."""
import torch
import torch.nn.functional as F
from neuralops.core.triton_kernels.elementwise import fused_swiglu, fused_fast_gelu


def test_fused_swiglu():
    x = torch.randn(2, 8)
    gate = torch.randn(2, 8)
    out = fused_swiglu(x, gate)
    assert torch.allclose(out, F.silu(gate) * x)


def test_fused_fast_gelu():
    x = torch.randn(2, 8)
    out = fused_fast_gelu(x)
    ref = F.gelu(x, approximate="tanh")
    assert torch.allclose(out, ref, atol=1e-4)
