#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Parity test for DeltaNet delta-rule recurrent scan Triton kernel against PyTorch reference.
"""

import pytest
import torch
import torch.nn.functional as F

from neuralops.ops.triton.delta_net_scan import delta_net_scan_pytorch, delta_net_scan_triton


@pytest.mark.gpu
def test_delta_net_scan_fp32_parity():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for Triton parity verification")

    torch.manual_seed(42)
    B, H, L, D_K, D_V = 2, 4, 64, 64, 64
    q = F.normalize(torch.randn(B, H, L, D_K, device="cuda", dtype=torch.float32), p=2, dim=-1)
    k = F.normalize(torch.randn(B, H, L, D_K, device="cuda", dtype=torch.float32), p=2, dim=-1)
    v = torch.randn(B, H, L, D_V, device="cuda", dtype=torch.float32)
    beta = torch.sigmoid(torch.randn(B, H, L, device="cuda", dtype=torch.float32))
    alpha = torch.sigmoid(torch.randn(B, H, L, device="cuda", dtype=torch.float32))

    ref = delta_net_scan_pytorch(q, k, v, beta=beta, alpha=alpha)
    tri = delta_net_scan_triton(q, k, v, beta=beta, alpha=alpha)

    assert torch.isfinite(tri).all(), "DeltaNet Triton output contains non-finite values"
    max_diff = (ref - tri).abs().max().item()
    assert max_diff < 1e-3, f"DeltaNet FP32 max diff {max_diff:.6e} exceeds 1e-3"


@pytest.mark.gpu
def test_delta_net_scan_fp16_parity():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for Triton parity verification")

    torch.manual_seed(42)
    B, H, L, D_K, D_V = 2, 4, 64, 64, 64
    q = F.normalize(torch.randn(B, H, L, D_K, device="cuda", dtype=torch.float16), p=2, dim=-1)
    k = F.normalize(torch.randn(B, H, L, D_K, device="cuda", dtype=torch.float16), p=2, dim=-1)
    v = torch.randn(B, H, L, D_V, device="cuda", dtype=torch.float16)
    beta = torch.sigmoid(torch.randn(B, H, L, device="cuda", dtype=torch.float16))
    alpha = torch.sigmoid(torch.randn(B, H, L, device="cuda", dtype=torch.float16))

    ref = delta_net_scan_pytorch(q, k, v, beta=beta, alpha=alpha)
    tri = delta_net_scan_triton(q, k, v, beta=beta, alpha=alpha)

    assert torch.isfinite(tri).all(), "DeltaNet FP16 Triton output contains non-finite values"
    max_diff = (ref.float() - tri.float()).abs().max().item()
    assert max_diff < 1e-2, f"DeltaNet FP16 max diff {max_diff:.6e} exceeds 1e-2"


@pytest.mark.gpu
def test_delta_net_scan_final_state_parity():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for Triton parity verification")

    torch.manual_seed(42)
    B, H, L, D_K, D_V = 2, 2, 64, 32, 32
    q = F.normalize(torch.randn(B, H, L, D_K, device="cuda", dtype=torch.float32), p=2, dim=-1)
    k = F.normalize(torch.randn(B, H, L, D_K, device="cuda", dtype=torch.float32), p=2, dim=-1)
    v = torch.randn(B, H, L, D_V, device="cuda", dtype=torch.float32)
    beta = torch.sigmoid(torch.randn(B, H, L, device="cuda", dtype=torch.float32))

    ref_out, ref_s = delta_net_scan_pytorch(q, k, v, beta=beta, return_final_state=True)
    tri_out, tri_s = delta_net_scan_triton(q, k, v, beta=beta, return_final_state=True)

    assert torch.isfinite(tri_out).all(), "DeltaNet output contains non-finite values"
    assert torch.isfinite(tri_s).all(), "DeltaNet final state contains non-finite values"

    max_diff_out = (ref_out - tri_out).abs().max().item()
    max_diff_s = (ref_s - tri_s).abs().max().item()
    assert max_diff_out < 1e-3, f"DeltaNet output max diff {max_diff_out:.6e} exceeds 1e-3"
    assert max_diff_s < 1e-3, f"DeltaNet final state max diff {max_diff_s:.6e} exceeds 1e-3"
