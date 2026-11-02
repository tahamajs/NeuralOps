#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Verification Suite for 3-Level Kernel Dispatch Hierarchy and Hardware Forcing.
Tests Level 1 (CUDA C++), Level 2 (Triton JIT), Level 3 (Vectorized PyTorch),
and Fail-Closed strictness preventing silent fallback.
"""

from __future__ import annotations

import pytest
import torch

from neuralops.ops.cuda._fallback import (
    dispatch,
    force_kernel_backend,
    get_dispatch_stats,
    reset_dispatch_stats,
    get_last_dispatch_backend,
)
from neuralops.ops.cuda import cosformer_linear_attention


@pytest.fixture(autouse=True)
def clean_dispatch_telemetry():
    """Reset dispatch statistics before each test."""
    reset_dispatch_stats()
    yield
    reset_dispatch_stats()


@pytest.mark.gpu
def test_dispatch_level3_pytorch_forcing():
    """Verify forcing 'pytorch' backend routes to PyTorch reference without GPU/Triton overhead."""
    device = torch.device("cuda:0")
    b, h, s, d = 2, 4, 32, 64
    q = torch.randn(b, h, s, d, device=device, dtype=torch.float32)
    k = torch.randn(b, h, s, d, device=device, dtype=torch.float32)
    v = torch.randn(b, h, s, d, device=device, dtype=torch.float32)

    with force_kernel_backend("pytorch"):
        out = cosformer_linear_attention(q, k, v)

    stats = get_dispatch_stats()
    assert stats["pytorch"] >= 1, f"Expected PyTorch dispatch counter increment, got {stats}"
    assert torch.isfinite(out).all().item(), "PyTorch forced fallback produced non-finite output."


@pytest.mark.gpu
def test_dispatch_level2_triton_forcing():
    """Verify forcing 'triton' backend executes Triton JIT or raises clear uncompiled error."""
    device = torch.device("cuda:0")
    b, h, s, d = 2, 4, 32, 64
    q = torch.randn(b, h, s, d, device=device, dtype=torch.float32)
    k = torch.randn(b, h, s, d, device=device, dtype=torch.float32)
    v = torch.randn(b, h, s, d, device=device, dtype=torch.float32)

    with force_kernel_backend("triton"):
        try:
            out = cosformer_linear_attention(q, k, v)
            assert torch.isfinite(out).all().item()
            backend = get_last_dispatch_backend("cosformer")
            assert backend == "triton", f"Expected 'triton' execution backend, got '{backend}'"
        except RuntimeError as exc:
            # Must explicitly fail-closed with Triton error, NOT silent fallback to CPU
            err_msg = str(exc)
            assert "Backend forced to 'triton'" in err_msg or "Triton" in err_msg, (
                f"Forcing Triton did not raise the expected fail-closed error: {err_msg}"
            )


@pytest.mark.gpu
def test_dispatch_level1_cuda_forcing():
    """Verify forcing 'cuda' backend executes native CUDA C++ kernel or raises fail-closed error."""
    device = torch.device("cuda:0")
    b, h, s, d = 2, 4, 32, 64
    q = torch.randn(b, h, s, d, device=device, dtype=torch.float32)
    k = torch.randn(b, h, s, d, device=device, dtype=torch.float32)
    v = torch.randn(b, h, s, d, device=device, dtype=torch.float32)

    with force_kernel_backend("cuda"):
        try:
            out = cosformer_linear_attention(q, k, v)
            assert torch.isfinite(out).all().item()
            backend = get_last_dispatch_backend("cosformer")
            assert backend == "cuda", f"Expected 'cuda' execution backend, got '{backend}'"
        except RuntimeError as exc:
            err_msg = str(exc)
            assert "Backend forced to 'cuda'" in err_msg or "CUDA" in err_msg, (
                f"Forcing CUDA did not raise expected fail-closed error: {err_msg}"
            )


def test_dispatch_fail_closed_cpu_tensor_when_cuda_forced():
    """
    CRITICAL INTEGRITY GATE:
    Passing a CPU tensor while forcing 'cuda' backend MUST raise a RuntimeError.
    Silent fallback to CPU or silent failure is strictly banned.
    """
    # Dummy operators
    def dummy_cuda_fn(x):
        if not x.is_cuda:
            raise ValueError("Expected CUDA tensor in native CUDA kernel")
        return x * 2

    def dummy_triton_fn(x):
        return x * 3

    def dummy_pytorch_ref(x):
        return x * 4

    cpu_tensor = torch.randn(4, 4, device="cpu")

    with force_kernel_backend("cuda"):
        with pytest.raises(RuntimeError) as exc_info:
            dispatch(
                dummy_cuda_fn,
                dummy_triton_fn,
                dummy_pytorch_ref,
                cpu_tensor,
                kernel_name="test_fail_closed_kernel",
            )
        assert "Backend forced to 'cuda'" in str(exc_info.value), (
            f"Expected explicit fail-closed message, got: {exc_info.value}"
        )


def test_dispatch_invalid_backend_name_raises():
    """Verify unknown backend names are immediately rejected."""
    with pytest.raises(ValueError) as exc_info:
        with force_kernel_backend("quantum_tensor_processor"):
            pass
    assert "Unknown backend" in str(exc_info.value)
