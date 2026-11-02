#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Parity test suite for 5 representative native CUDA C++ kernels against PyTorch reference.
Verifies layernorm, rmsnorm, fast_gelu, swiglu, and ternary_gemm with max_diff < 1e-4.
"""

from __future__ import annotations

import pytest
import torch

from neuralops.ops.cuda._fallback import force_kernel_backend
from neuralops.ops.cuda.fast_gelu.fast_gelu_autograd import fast_gelu, fast_gelu_pytorch_ref
from neuralops.ops.cuda.layernorm.layernorm_autograd import layernorm, layernorm_pytorch_ref
from neuralops.ops.cuda.rmsnorm.rmsnorm_autograd import rmsnorm, rmsnorm_pytorch_ref
from neuralops.ops.cuda.swiglu.swiglu_autograd import swiglu, swiglu_pytorch_ref
from neuralops.ops.cuda.ternary_gemm.ternary_autograd import ternary_gemm, ternary_gemm_pytorch_ref


@pytest.mark.gpu
def test_layernorm_cuda_parity():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for native CUDA parity tests")

    torch.manual_seed(42)
    B, T, D = 4, 32, 128
    x = torch.randn(B, T, D, device="cuda", dtype=torch.float32)
    weight = torch.ones(D, device="cuda", dtype=torch.float32)
    bias = torch.randn(D, device="cuda", dtype=torch.float32)
    residual = torch.randn(B, T, D, device="cuda", dtype=torch.float32)

    ref = layernorm_pytorch_ref(x, weight, bias=bias, residual=residual, eps=1e-5)
    with force_kernel_backend("cuda"):
        cuda_out = layernorm(x, weight, bias=bias, residual=residual, eps=1e-5)

    assert torch.isfinite(cuda_out).all(), "CUDA layernorm produced non-finite values"
    max_diff = (ref - cuda_out).abs().max().item()
    assert max_diff < 1e-4, f"CUDA layernorm max diff {max_diff:.6e} exceeds 1e-4"


@pytest.mark.gpu
def test_rmsnorm_cuda_parity():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for native CUDA parity tests")

    torch.manual_seed(42)
    B, T, D = 4, 32, 128
    x = torch.randn(B, T, D, device="cuda", dtype=torch.float32)
    weight = torch.ones(D, device="cuda", dtype=torch.float32)
    residual = torch.randn(B, T, D, device="cuda", dtype=torch.float32)

    ref = rmsnorm_pytorch_ref(x, weight, residual=residual, eps=1e-5)
    with force_kernel_backend("cuda"):
        cuda_out = rmsnorm(x, weight, residual=residual, eps=1e-5)

    assert torch.isfinite(cuda_out).all(), "CUDA rmsnorm produced non-finite values"
    max_diff = (ref - cuda_out).abs().max().item()
    assert max_diff < 1e-4, f"CUDA rmsnorm max diff {max_diff:.6e} exceeds 1e-4"


@pytest.mark.gpu
def test_fast_gelu_cuda_parity():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for native CUDA parity tests")

    torch.manual_seed(42)
    x = torch.randn(8, 64, 128, device="cuda", dtype=torch.float32)

    ref = fast_gelu_pytorch_ref(x)
    with force_kernel_backend("cuda"):
        cuda_out = fast_gelu(x)

    assert torch.isfinite(cuda_out).all(), "CUDA fast_gelu produced non-finite values"
    max_diff = (ref - cuda_out).abs().max().item()
    assert max_diff < 1e-4, f"CUDA fast_gelu max diff {max_diff:.6e} exceeds 1e-4"


@pytest.mark.gpu
def test_swiglu_cuda_parity():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for native CUDA parity tests")

    torch.manual_seed(42)
    B, T, D = 4, 32, 256
    x = torch.randn(B, T, D, device="cuda", dtype=torch.float32)

    ref = swiglu_pytorch_ref(x)
    with force_kernel_backend("cuda"):
        cuda_out = swiglu(x)

    assert torch.isfinite(cuda_out).all(), "CUDA swiglu produced non-finite values"
    max_diff = (ref - cuda_out).abs().max().item()
    assert max_diff < 1e-4, f"CUDA swiglu max diff {max_diff:.6e} exceeds 1e-4"


@pytest.mark.gpu
def test_ternary_gemm_cuda_parity():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for native CUDA parity tests")

    torch.manual_seed(42)
    M, K, N = 16, 64, 32
    x = torch.randn(M, K, device="cuda", dtype=torch.float32)
    # w_packed: shape (N, K // 4) int8
    w_packed = torch.randint(0, 256, (N, K // 4), device="cuda", dtype=torch.uint8)
    scales = torch.rand(N, device="cuda", dtype=torch.float32) * 0.1 + 0.01
    bias = torch.randn(N, device="cuda", dtype=torch.float32)

    ref = ternary_gemm_pytorch_ref(x, w_packed, scales, bias=bias)
    with force_kernel_backend("cuda"):
        cuda_out = ternary_gemm(x, w_packed, scales, bias=bias)

    assert torch.isfinite(cuda_out).all(), "CUDA ternary_gemm produced non-finite values"
    max_diff = (ref - cuda_out).abs().max().item()
    assert max_diff < 1e-4, f"CUDA ternary_gemm max diff {max_diff:.6e} exceeds 1e-4"
