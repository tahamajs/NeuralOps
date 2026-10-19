#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Parity tests for Quantized GEMM (Module 3).
"""

import pytest
import torch

from neuralops.core.cuda_kernels.quantized_gemm import (
    w4a16_gemm,
    w4a16_gemm_pytorch_ref,
    fp8_gemm,
    fp8_gemm_pytorch_ref,
)


def test_w4a16_gemm_forward_parity():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for Quantized GEMM test")

    torch.manual_seed(42)
    M, K, N = 16, 64, 32
    group_size = 32

    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    W_q = torch.randint(0, 255, (K, N // 2), device="cuda", dtype=torch.uint8)
    scales = torch.rand(K // group_size, N, device="cuda", dtype=torch.float32) * 0.1
    zeros = torch.randint(0, 8, (K // group_size, N), device="cuda", dtype=torch.float32)

    out_cuda = w4a16_gemm(A, W_q, scales, zeros, group_size=group_size)
    out_ref = w4a16_gemm_pytorch_ref(A, W_q, scales, zeros, group_size=group_size)

    torch.testing.assert_close(out_cuda, out_ref, atol=1e-3, rtol=1e-3)


def test_fp8_gemm_forward_parity():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for Quantized GEMM test")

    torch.manual_seed(42)
    M, K, N = 16, 64, 32

    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, N, device="cuda", dtype=torch.float32)
    scale_a = torch.tensor(0.5, device="cuda", dtype=torch.float32)
    scale_b = torch.tensor(0.25, device="cuda", dtype=torch.float32)

    out_cuda = fp8_gemm(A, B, scale_a, scale_b)
    out_ref = fp8_gemm_pytorch_ref(A, B, scale_a, scale_b)

    torch.testing.assert_close(out_cuda, out_ref, atol=1e-3, rtol=1e-3)
