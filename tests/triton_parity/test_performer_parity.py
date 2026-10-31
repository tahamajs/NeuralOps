#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Parity test: Performer FAVOR+ Triton Kernel vs PyTorch Reference
"""

import pytest
import torch

from neuralops.ops.triton.performer_triton import (
    performer_pytorch_reference,
    performer_triton_forward,
)


@pytest.mark.gpu
def test_performer_triton_matches_pytorch():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required for Triton kernel verification")

    device = torch.device("cuda")
    B, H, L, M_DIM, D = 2, 4, 64, 32, 64

    torch.manual_seed(42)
    # Ensure positive feature projections
    q_prime = torch.abs(torch.randn(B, H, L, M_DIM, device=device, dtype=torch.float32)) + 0.1
    k_prime = torch.abs(torch.randn(B, H, L, M_DIM, device=device, dtype=torch.float32)) + 0.1
    v = torch.randn(B, H, L, D, device=device, dtype=torch.float32)

    ref_out = performer_pytorch_reference(q_prime, k_prime, v)
    tri_out = performer_triton_forward(q_prime, k_prime, v)

    torch.testing.assert_close(tri_out, ref_out, atol=1e-3, rtol=1e-3)
