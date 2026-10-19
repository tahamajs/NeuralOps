# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Parity tests for Sliding Window Attention with Attention Sinks (Module 7).
"""

import pytest
import torch

from neuralops.core.cuda_kernels.sliding_window import (
    sliding_window_attn,
    sliding_window_pytorch_ref,
)


def test_sliding_window_parity():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for Sliding Window test")

    torch.manual_seed(42)
    B, H, N, D = 2, 4, 64, 64
    window_size = 16
    num_sinks = 4

    q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)
    k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)
    v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)

    out_cuda = sliding_window_attn(q, k, v, window_size=window_size, num_sinks=num_sinks)
    out_ref = sliding_window_pytorch_ref(q, k, v, window_size=window_size, num_sinks=num_sinks)

    torch.testing.assert_close(out_cuda, out_ref, atol=1e-3, rtol=1e-3)
