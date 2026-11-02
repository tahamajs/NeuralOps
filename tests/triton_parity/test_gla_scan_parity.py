#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Parity test for Chunked Gated Linear Attention (GLA) Triton kernel against PyTorch reference.
"""

import pytest
import torch

from neuralops.ops.triton.gla_scan import (
    chunked_gla_scan_pytorch,
    chunked_gla_scan_triton,
    gla_scan_pytorch,
    gla_scan_triton,
)


@pytest.mark.gpu
def test_chunked_gla_scan_fp32_parity():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for Triton parity verification")

    torch.manual_seed(42)
    B, H, L, D = 2, 4, 128, 64
    q = torch.randn(B, H, L, D, device="cuda", dtype=torch.float32)
    k = torch.randn(B, H, L, D, device="cuda", dtype=torch.float32)
    v = torch.randn(B, H, L, D, device="cuda", dtype=torch.float32)
    lambdas = torch.sigmoid(torch.randn(B, H, L, device="cuda", dtype=torch.float32)).clamp(1e-4, 1.0 - 1e-4)

    ref = chunked_gla_scan_pytorch(q, k, v, lambdas=lambdas, chunk_size=64)
    tri = chunked_gla_scan_triton(q, k, v, lambdas=lambdas, chunk_size=64)

    assert torch.isfinite(tri).all(), "Chunked GLA Triton output contains non-finite values"
    max_diff = (ref - tri).abs().max().item()
    assert max_diff < 1e-3, f"Chunked GLA FP32 max diff {max_diff:.6e} exceeds 1e-3"


@pytest.mark.gpu
def test_chunked_gla_scan_fp16_parity():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for Triton parity verification")

    torch.manual_seed(42)
    B, H, L, D = 2, 4, 128, 64
    q = torch.randn(B, H, L, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, H, L, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, H, L, D, device="cuda", dtype=torch.float16)
    lambdas = torch.sigmoid(torch.randn(B, H, L, device="cuda", dtype=torch.float16)).clamp(1e-4, 1.0 - 1e-4)

    ref = chunked_gla_scan_pytorch(q, k, v, lambdas=lambdas, chunk_size=64)
    tri = chunked_gla_scan_triton(q, k, v, lambdas=lambdas, chunk_size=64)

    assert torch.isfinite(tri).all(), "Chunked GLA FP16 Triton output contains non-finite values"
    max_diff = (ref.float() - tri.float()).abs().max().item()
    assert max_diff < 1e-2, f"Chunked GLA FP16 max diff {max_diff:.6e} exceeds 1e-2"


@pytest.mark.gpu
def test_recurrent_gla_scan_parity():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for Triton parity verification")

    torch.manual_seed(42)
    B, H, L, D = 2, 2, 64, 32
    q = torch.randn(B, H, L, D, device="cuda", dtype=torch.float32)
    k = torch.randn(B, H, L, D, device="cuda", dtype=torch.float32)
    v = torch.randn(B, H, L, D, device="cuda", dtype=torch.float32)
    lambdas = torch.sigmoid(torch.randn(B, H, L, device="cuda", dtype=torch.float32)).clamp(1e-4, 1.0 - 1e-4)

    ref = gla_scan_pytorch(q, k, v, lambdas=lambdas, use_normalizer=False)
    tri = gla_scan_triton(q, k, v, lambdas=lambdas, use_normalizer=False)

    assert torch.isfinite(tri).all(), "Recurrent GLA Triton output contains non-finite values"
    max_diff = (ref - tri).abs().max().item()
    assert max_diff < 1e-3, f"Recurrent GLA FP32 max diff {max_diff:.6e} exceeds 1e-3"
