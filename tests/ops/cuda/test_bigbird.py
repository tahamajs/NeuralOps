#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Unit and regression tests for BigBird Sparse Attention CUDA operator and fallback dispatch."""

from __future__ import annotations

import pytest
import torch

from neuralops.ops.cuda._fallback import (
    force_kernel_backend,
    get_last_dispatch_backend,
)
from neuralops.ops.cuda.bigbird import bigbird, bigbird_ref, bigbird_triton_ref


def test_bigbird_symbol_export_and_callable() -> None:
    """Verify that bigbird symbols are properly exported and callable."""
    assert callable(bigbird)
    assert callable(bigbird_ref)
    assert callable(bigbird_triton_ref)


def test_bigbird_basic_cpu() -> None:
    """Test basic BigBird attention computation on CPU."""
    torch.manual_seed(42)
    B, H, N, D = 2, 4, 32, 16
    q = torch.randn(B, H, N, D)
    k = torch.randn(B, H, N, D)
    v = torch.randn(B, H, N, D)

    out = bigbird(q, k, v, window_size=8, num_global_tokens=2, num_random_blocks=1)
    ref = bigbird_ref(q, k, v, window_size=8, num_global_tokens=2, num_random_blocks=1)

    assert out.shape == (B, H, N, D)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_bigbird_zero_dim_shapes() -> None:
    """Test empty / zero-dimension tensor handling."""
    q = torch.randn(0, 4, 32, 16)
    k = torch.randn(0, 4, 32, 16)
    v = torch.randn(0, 4, 32, 16)
    out = bigbird(q, k, v)
    assert out.shape == (0, 4, 32, 16)


def test_bigbird_invalid_shapes() -> None:
    """Test validation errors on rank or shape mismatch."""
    # 3D instead of 4D
    q3 = torch.randn(2, 32, 16)
    k3 = torch.randn(2, 32, 16)
    v3 = torch.randn(2, 32, 16)
    with pytest.raises(ValueError, match="4D tensors"):
        bigbird(q3, k3, v3)

    # Shape mismatch
    q4 = torch.randn(2, 4, 32, 16)
    k4 = torch.randn(2, 4, 16, 16)
    v4 = torch.randn(2, 4, 32, 16)
    with pytest.raises(ValueError, match="Shape mismatch"):
        bigbird(q4, k4, v4)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_bigbird_dtypes(dtype: torch.dtype) -> None:
    """Test BigBird attention across multiple floating point dtypes."""
    torch.manual_seed(42)
    B, H, N, D = 1, 2, 16, 8
    q = torch.randn(B, H, N, D, dtype=dtype)
    k = torch.randn(B, H, N, D, dtype=dtype)
    v = torch.randn(B, H, N, D, dtype=dtype)

    out = bigbird(q, k, v, window_size=4, num_global_tokens=2, num_random_blocks=1)
    ref = bigbird_ref(q, k, v, window_size=4, num_global_tokens=2, num_random_blocks=1)

    assert out.shape == (B, H, N, D)
    assert out.dtype == dtype
    assert torch.isfinite(out).all()
    # Parity check
    tol = 1e-2 if dtype != torch.float32 else 1e-5
    assert torch.allclose(out, ref, atol=tol, rtol=tol)


def test_bigbird_amp_cpu() -> None:
    """Verify torch.amp autocast compliance without deprecated torch.cuda.amp."""
    torch.manual_seed(42)
    B, H, N, D = 2, 2, 16, 16
    q = torch.randn(B, H, N, D)
    k = torch.randn(B, H, N, D)
    v = torch.randn(B, H, N, D)

    # Use torch.amp.autocast in compliance with Rule 2.1
    with torch.amp.autocast("cpu", dtype=torch.bfloat16):
        out = bigbird(q, k, v, window_size=4, num_global_tokens=2, num_random_blocks=1)

    assert out.shape == (B, H, N, D)
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out).all()


def test_bigbird_autograd_gradient_flow() -> None:
    """Verify autograd backward produces valid gradients for q, k, and v."""
    torch.manual_seed(42)
    B, H, N, D = 2, 2, 16, 8
    q = torch.randn(B, H, N, D, requires_grad=True)
    k = torch.randn(B, H, N, D, requires_grad=True)
    v = torch.randn(B, H, N, D, requires_grad=True)

    out = bigbird(q, k, v, window_size=4, num_global_tokens=2, num_random_blocks=1)
    loss = out.sum()
    loss.backward()

    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None
    assert q.grad.shape == q.shape
    assert k.grad.shape == k.shape
    assert v.grad.shape == v.shape
    assert torch.isfinite(q.grad).all()
    assert torch.isfinite(k.grad).all()
    assert torch.isfinite(v.grad).all()


def test_bigbird_force_pytorch_backend() -> None:
    """Verify fallback dispatch telemetry when forcing PyTorch backend."""
    torch.manual_seed(42)
    B, H, N, D = 1, 2, 16, 8
    q = torch.randn(B, H, N, D)
    k = torch.randn(B, H, N, D)
    v = torch.randn(B, H, N, D)

    with force_kernel_backend("pytorch"):
        out = bigbird(q, k, v, window_size=4, num_global_tokens=2, num_random_blocks=1)
        assert out.shape == (B, H, N, D)
        backend = get_last_dispatch_backend("bigbird")
        assert backend == "pytorch"


@pytest.mark.parametrize("window_size", [2, 8, 32])
@pytest.mark.parametrize("num_global_tokens", [0, 2, 4])
def test_bigbird_parameter_variations(window_size: int, num_global_tokens: int) -> None:
    """Verify attention patterns across varying window sizes and global token counts."""
    torch.manual_seed(42)
    B, H, N, D = 1, 2, 32, 16
    q = torch.randn(B, H, N, D)
    k = torch.randn(B, H, N, D)
    v = torch.randn(B, H, N, D)

    out = bigbird(
        q,
        k,
        v,
        window_size=window_size,
        num_global_tokens=num_global_tokens,
        num_random_blocks=2,
    )
    assert out.shape == (B, H, N, D)
    assert torch.isfinite(out).all()


@pytest.mark.gpu
def test_bigbird_cuda_parity() -> None:
    """Verify CUDA kernel forward parity against PyTorch reference when CUDA is available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable on host")

    torch.manual_seed(42)
    B, H, N, D = 2, 4, 64, 32
    device = torch.device("cuda")
    q = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
    k = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
    v = torch.randn(B, H, N, D, device=device, dtype=torch.float32)

    with torch.amp.autocast("cuda", dtype=torch.float16):
        out = bigbird(q, k, v, window_size=16, num_global_tokens=4, num_random_blocks=2)
        ref = bigbird_ref(q, k, v, window_size=16, num_global_tokens=4, num_random_blocks=2)

    diff = (out.float() - ref.float()).abs().max().item()
    assert diff < 1e-2, f"Max diff vs ref too large: {diff}"
    assert torch.isfinite(out).all()
