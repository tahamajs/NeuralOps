#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Parity test: NASC Attention Triton Kernel vs PyTorch Reference
"""

import pytest
import torch
import torch.nn.functional as F

from neuralops.ops.triton.nasc_triton import (
    nasc_pytorch_reference,
    nasc_triton_forward,
)


@pytest.mark.gpu
def test_nasc_triton_matches_pytorch():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required for Triton kernel verification")

    device = torch.device("cuda")
    B, H, L, D = 2, 4, 128, 32

    torch.manual_seed(42)
    o1 = torch.randn(B, H, L, D, device=device, dtype=torch.float32)
    o2 = torch.randn(B, H, L, D, device=device, dtype=torch.float32)
    o3 = torch.randn(B, H, L, D, device=device, dtype=torch.float32)
    gates = F.softmax(torch.randn(B, H, L, 3, device=device, dtype=torch.float32), dim=-1)

    ref_out = nasc_pytorch_reference(o1, o2, o3, gates)
    tri_out = nasc_triton_forward(o1, o2, o3, gates)

    torch.testing.assert_close(tri_out, ref_out, atol=1e-3, rtol=1e-3)
