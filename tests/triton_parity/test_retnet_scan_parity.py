#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Parity test for RetNet multi-scale retention Triton kernel against PyTorch reference.
"""

import pytest
import torch

from neuralops.ops.triton.retnet_scan import retnet_scan_pytorch, retnet_scan_triton


@pytest.mark.gpu
def test_retnet_scan_fp32_parity():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for Triton parity verification")

    torch.manual_seed(42)
    B, H, L, D = 2, 4, 128, 64
    q = torch.randn(B, H, L, D, device="cuda", dtype=torch.float32)
    k = torch.randn(B, H, L, D, device="cuda", dtype=torch.float32)
    v = torch.randn(B, H, L, D, device="cuda", dtype=torch.float32)
    gamma = torch.tensor([0.9, 0.95, 0.85, 0.99], device="cuda", dtype=torch.float32)

    ref = retnet_scan_pytorch(q, k, v, gamma=gamma)
    tri = retnet_scan_triton(q, k, v, gamma=gamma)

    assert torch.isfinite(tri).all(), "RetNet Triton output contains non-finite values"
    max_diff = (ref - tri).abs().max().item()
    assert max_diff < 1e-3, f"RetNet FP32 max diff {max_diff:.6e} exceeds 1e-3"


@pytest.mark.gpu
def test_retnet_scan_fp16_parity():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for Triton parity verification")

    torch.manual_seed(42)
    B, H, L, D = 2, 4, 64, 64
    scale = D ** -0.5
    q = torch.randn(B, H, L, D, device="cuda", dtype=torch.float16) * scale
    k = torch.randn(B, H, L, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, H, L, D, device="cuda", dtype=torch.float16)
    gamma = torch.tensor([0.9, 0.95, 0.85, 0.99], device="cuda", dtype=torch.float32)

    ref = retnet_scan_pytorch(q, k, v, gamma=gamma)
    tri = retnet_scan_triton(q, k, v, gamma=gamma)

    assert torch.isfinite(tri).all(), "RetNet FP16 Triton output contains non-finite values"
    max_diff = (ref.float() - tri.float()).abs().max().item()
    assert max_diff < 1e-2, f"RetNet FP16 max diff {max_diff:.6e} exceeds 1e-2"


@pytest.mark.gpu
def test_retnet_scan_final_state_parity():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for Triton parity verification")

    torch.manual_seed(42)
    B, H, L, D = 2, 2, 64, 32
    q = torch.randn(B, H, L, D, device="cuda", dtype=torch.float32)
    k = torch.randn(B, H, L, D, device="cuda", dtype=torch.float32)
    v = torch.randn(B, H, L, D, device="cuda", dtype=torch.float32)
    gamma = 0.92

    ref_out, ref_s = retnet_scan_pytorch(q, k, v, gamma=gamma, return_final_state=True)
    tri_out, tri_s = retnet_scan_triton(q, k, v, gamma=gamma, return_final_state=True)

    assert torch.isfinite(tri_out).all(), "RetNet output contains non-finite values"
    assert torch.isfinite(tri_s).all(), "RetNet final state contains non-finite values"

    max_diff_out = (ref_out - tri_out).abs().max().item()
    max_diff_s = (ref_s - tri_s).abs().max().item()
    assert max_diff_out < 1e-3, f"RetNet output max diff {max_diff_out:.6e} exceeds 1e-3"
    assert max_diff_s < 1e-3, f"RetNet final state max diff {max_diff_s:.6e} exceeds 1e-3"
