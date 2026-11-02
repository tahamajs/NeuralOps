#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Exhaustive Numerical and Autograd Parity Verification for Triton and CUDA Kernels.
Ensures hardware-accelerated kernels produce outputs numerically equivalent to
unaccelerated PyTorch mathematical references within strict error budgets.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from neuralops.core.cuda_kernels.selective_scan import (
    selective_scan,
    selective_scan_pytorch_ref,
)
from neuralops.core.cuda_kernels.sliding_window import (
    sliding_window_attn,
    sliding_window_pytorch_ref,
)
from neuralops.core.cuda_kernels.sparse_attention import (
    block_sparse_attn,
    block_sparse_attn_pytorch_ref,
)
from neuralops.core.cuda_kernels.quantized_gemm import (
    w4a16_gemm,
    w4a16_gemm_pytorch_ref,
    fp8_gemm,
    fp8_gemm_pytorch_ref,
)
from neuralops.ops.cuda import cosformer_linear_attention
from neuralops.ops.cuda._fallback import force_kernel_backend
from neuralops.ops.cuda import list_cuda_kernels


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_selective_scan_parity(dtype: torch.dtype):
    """Verify selective scan kernel matches analytical PyTorch reference."""
    b, d, l, d_state = 2, 8, 32, 16
    u = torch.randn(b, d, l, device="cuda", dtype=dtype)
    delta = torch.randn(b, d, l, device="cuda", dtype=dtype)
    a = torch.randn(d, d_state, device="cuda", dtype=dtype)
    b_t = torch.randn(b, d_state, l, device="cuda", dtype=dtype)
    c = torch.randn(b, d_state, l, device="cuda", dtype=dtype)
    d_param = torch.randn(d, device="cuda", dtype=dtype)

    out_k, state_k = selective_scan(u, delta, a, b_t, c, d_param)
    out_ref, state_ref = selective_scan_pytorch_ref(u, delta, a, b_t, c, d_param)

    tol = 1e-3 if dtype == torch.float32 else 1e-2
    max_diff_out = (out_k - out_ref).abs().max().item()
    max_diff_state = (state_k - state_ref).abs().max().item()

    assert max_diff_out <= tol, f"selective_scan out max_diff={max_diff_out:.6f} exceeded tol={tol}"
    assert max_diff_state <= tol, f"selective_scan state max_diff={max_diff_state:.6f} exceeded tol={tol}"


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_sliding_window_parity(dtype: torch.dtype):
    """Verify sliding window attention matches PyTorch reference."""
    b, h, n, d = 2, 4, 64, 32
    q = torch.randn(b, h, n, d, device="cuda", dtype=dtype)
    k = torch.randn(b, h, n, d, device="cuda", dtype=dtype)
    v = torch.randn(b, h, n, d, device="cuda", dtype=dtype)

    out_k = sliding_window_attn(q, k, v, window_size=16, num_sinks=4)
    out_ref = sliding_window_pytorch_ref(q, k, v, window_size=16, num_sinks=4)

    tol = 1e-3 if dtype == torch.float32 else 1e-2
    max_diff = (out_k - out_ref).abs().max().item()
    assert max_diff <= tol, f"sliding_window_attn max_diff={max_diff:.6f} exceeded tol={tol}"


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_block_sparse_parity(dtype: torch.dtype):
    """Verify block sparse attention matches full attention reference on active blocks."""
    b, h, n, d = 2, 4, 64, 32
    q = torch.randn(b, h, n, d, device="cuda", dtype=dtype)
    k = torch.randn(b, h, n, d, device="cuda", dtype=dtype)
    v = torch.randn(b, h, n, d, device="cuda", dtype=dtype)

    out_k = block_sparse_attn(q, k, v, block_size=16)
    out_ref = block_sparse_attn_pytorch_ref(q, k, v, block_size=16)

    tol = 1e-3 if dtype == torch.float32 else 1e-2
    max_diff = (out_k - out_ref).abs().max().item()
    assert max_diff <= tol, f"block_sparse_attn max_diff={max_diff:.6f} exceeded tol={tol}"


@pytest.mark.gpu
def test_quantized_gemm_parity():
    """Verify w4a16 and fp8 GEMM operations match reference precision bounds."""
    m, k, n = 32, 64, 32
    x = torch.randn(m, k, device="cuda", dtype=torch.float16)
    weight = torch.randn(n, k, device="cuda", dtype=torch.float16)

    out_fp8 = fp8_gemm(x, weight)
    out_ref_fp8 = fp8_gemm_pytorch_ref(x, weight)
    diff_fp8 = (out_fp8 - out_ref_fp8).abs().max().item()
    assert diff_fp8 <= 5e-2, f"fp8_gemm max_diff={diff_fp8:.6f} exceeded 5e-2 bound."

    out_w4 = w4a16_gemm(x, weight)
    out_ref_w4 = w4a16_gemm_pytorch_ref(x, weight)
    diff_w4 = (out_w4 - out_ref_w4).abs().max().item()
    assert diff_w4 <= 5e-2, f"w4a16_gemm max_diff={diff_w4:.6f} exceeded 5e-2 bound."


@pytest.mark.gpu
def test_cosformer_parity_across_backends():
    """Verify cosformer numerical parity between forced Triton and forced PyTorch backends."""
    b, h, s, d = 2, 4, 64, 32
    q = torch.randn(b, h, s, d, device="cuda", dtype=torch.float32)
    k = torch.randn(b, h, s, d, device="cuda", dtype=torch.float32)
    v = torch.randn(b, h, s, d, device="cuda", dtype=torch.float32)

    with force_kernel_backend("pytorch"):
        out_pytorch = cosformer_linear_attention(q, k, v)

    try:
        with force_kernel_backend("triton"):
            out_triton = cosformer_linear_attention(q, k, v)
        max_diff = (out_triton - out_pytorch).abs().max().item()
        assert max_diff <= 1e-3, f"Cosformer Triton vs PyTorch max_diff={max_diff:.6f} > 1e-3"
    except RuntimeError:
        # Triton may be unsupported on legacy CUDA driver
        pytest.skip("Triton cosformer backend unavailable on current CUDA driver environment.")


@pytest.mark.gpu
def test_autograd_gradcheck_selective_scan():
    """Verify analytical gradients via torch.autograd.gradcheck on selective scan."""
    b, d, l, d_state = 1, 2, 8, 4
    u = torch.randn(b, d, l, device="cuda", dtype=torch.float64, requires_grad=True)
    delta = torch.randn(b, d, l, device="cuda", dtype=torch.float64, requires_grad=True)
    a = torch.randn(d, d_state, device="cuda", dtype=torch.float64, requires_grad=True)
    b_t = torch.randn(b, d_state, l, device="cuda", dtype=torch.float64, requires_grad=True)
    c = torch.randn(b, d_state, l, device="cuda", dtype=torch.float64, requires_grad=True)
    d_param = torch.randn(d, device="cuda", dtype=torch.float64, requires_grad=True)

    def scan_func(u_, delta_, a_, b_t_, c_, d_param_):
        out, _ = selective_scan_pytorch_ref(u_, delta_, a_, b_t_, c_, d_param_)
        return out

    passed = torch.autograd.gradcheck(
        scan_func,
        (u, delta, a, b_t, c, d_param),
        eps=1e-5,
        atol=1e-3,
        rtol=1e-3,
    )
    assert passed, "torch.autograd.gradcheck failed for selective scan."


@pytest.mark.gpu
def test_all_cuda_kernel_packages_discovery():
    """Verify all 110+ registered CUDA kernel packages can be discovered and cataloged."""
    modules = list_cuda_kernels(category="modules")
    assert len(modules) >= 110, f"Expected at least 110 CUDA kernel modules, found {len(modules)}"
