#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Parity test: Tensor-Product Attention Triton Kernel vs PyTorch Reference
"""

import pytest
import torch

from neuralops.ops.triton.tpa_triton import (
    tpa_pytorch_reference,
    tpa_triton_forward,
)


@pytest.mark.gpu
def test_tpa_triton_matches_pytorch():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required for Triton kernel verification")

    device = torch.device("cuda")
    B, H, L, RA, RB = 2, 4, 128, 8, 8

    torch.manual_seed(42)
    q1 = torch.randn(B, H, L, RA, device=device, dtype=torch.float32)
    q2 = torch.randn(B, H, L, RB, device=device, dtype=torch.float32)

    ref_out = tpa_pytorch_reference(q1, q2)
    tri_out = tpa_triton_forward(q1, q2)

    torch.testing.assert_close(tri_out, ref_out, atol=1e-3, rtol=1e-3)
