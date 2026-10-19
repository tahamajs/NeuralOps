#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for Sliding Window Triton kernel and fallback."""
import torch
from neuralops.core.triton_kernels.sliding_window import sliding_window_attention_kernel, sliding_window_pytorch


def test_sliding_window_kernel_matches_reference():
    torch.manual_seed(42)
    B, H, N, D = 2, 2, 64, 32
    q = torch.randn(B, H, N, D)
    k = torch.randn(B, H, N, D)
    v = torch.randn(B, H, N, D)

    ref = sliding_window_pytorch(q, k, v, window_size=16)
    out = sliding_window_attention_kernel(q, k, v, window_size=16)

    assert torch.allclose(ref, out, atol=1e-5)
    assert out.shape == (B, H, N, D)
