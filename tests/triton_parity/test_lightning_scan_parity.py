#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Parity test for Lightning Attention Scan Triton kernel against PyTorch reference.
"""

import pytest
import torch

from neuralops.ops.triton.lightning_scan import (
    lightning_scan_pytorch,
    lightning_scan,
)


@pytest.mark.gpu
def test_lightning_scan_fp32_parity():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for Triton parity verification")

    torch.manual_seed(42)
    B, H, L, D = 2, 4, 128, 64
    q = torch.randn(B, H, L, D, device="cuda", dtype=torch.float32)
    k = torch.randn(B, H, L, D, device="cuda", dtype=torch.float32)
    v = torch.randn(B, H, L, D, device="cuda", dtype=torch.float32)
    slope = torch.linspace(0.1, 0.9, H, device="cuda", dtype=torch.float32)

    ref = lightning_scan_pytorch(q, k, v, slope=slope)
    tri = lightning_scan(q, k, v, slope=slope)

    assert torch.isfinite(tri).all(), "Lightning Scan Triton output contains non-finite values"
    max_diff = (ref - tri).abs().max().item()
    assert max_diff < 1e-3, f"Lightning Scan FP32 max diff {max_diff:.6e} exceeds 1e-3"


@pytest.mark.gpu
def test_lightning_scan_fp16_parity():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for Triton parity verification")

    torch.manual_seed(42)
    B, H, L, D = 2, 4, 128, 64
    q = torch.randn(B, H, L, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, H, L, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, H, L, D, device="cuda", dtype=torch.float16)
    slope = torch.linspace(0.1, 0.9, H, device="cuda", dtype=torch.float16)

    ref = lightning_scan_pytorch(q, k, v, slope=slope)
    tri = lightning_scan(q, k, v, slope=slope)

    assert torch.isfinite(tri).all(), "Lightning Scan Triton FP16 output contains non-finite values"
    max_diff = (ref - tri).abs().max().item()
    assert max_diff < 2e-2, f"Lightning Scan FP16 max diff {max_diff:.6e} exceeds 2e-2"
