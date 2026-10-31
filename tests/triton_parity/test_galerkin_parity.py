#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Parity test: Galerkin Attention Triton Kernel vs PyTorch Reference
"""

import pytest
import torch

from neuralops.ops.triton.galerkin_triton import (
    galerkin_pytorch_reference,
    galerkin_triton_forward,
)


@pytest.mark.gpu
def test_galerkin_triton_matches_pytorch():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required for Triton kernel verification")

    device = torch.device("cuda")
    B, H, L, D = 2, 4, 128, 32

    torch.manual_seed(42)
    q = torch.randn(B, H, L, D, device=device, dtype=torch.float32)
    k = torch.randn(B, H, L, D, device=device, dtype=torch.float32)
    v = torch.randn(B, H, L, D, device=device, dtype=torch.float32)

    ref_out = galerkin_pytorch_reference(q, k, v)
    tri_out = galerkin_triton_forward(q, k, v)

    torch.testing.assert_close(tri_out, ref_out, atol=1e-3, rtol=1e-3)
