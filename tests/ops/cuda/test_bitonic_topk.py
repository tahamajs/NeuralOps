#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Unit and regression tests for Bitonic Top-K CUDA operator and fallback dispatch."""

from __future__ import annotations

import pytest
import torch

from neuralops.ops.cuda._fallback import force_kernel_backend
from neuralops.ops.cuda.bitonic_topk import bitonic_topk, bitonic_topk_pytorch_ref


def test_bitonic_topk_basic_cpu() -> None:
    """Test basic top-k selection on CPU with float32."""
    x = torch.tensor([[1.0, 5.0, 2.0, 8.0, 3.0], [9.0, 4.0, 7.0, 1.0, 6.0]])
    k = 3
    indices = bitonic_topk(x, k)
    ref = bitonic_topk_pytorch_ref(x, k)

    assert indices.shape == (2, 3)
    assert indices.dtype == torch.int64
    assert torch.equal(indices, ref)


def test_bitonic_topk_boundary_k() -> None:
    """Test boundary values: K=0, K=N, and full row."""
    x = torch.randn(4, 16)

    # K = 0
    out_k0 = bitonic_topk(x, 0)
    assert out_k0.shape == (4, 0)
    assert out_k0.dtype == torch.int64

    # K = N
    out_kn = bitonic_topk(x, 16)
    ref_kn = bitonic_topk_pytorch_ref(x, 16)
    assert out_kn.shape == (4, 16)
    assert torch.equal(out_kn, ref_kn)


def test_bitonic_topk_invalid_k() -> None:
    """Test negative K and K > N raises ValueError."""
    x = torch.randn(4, 16)

    with pytest.raises(ValueError, match="K must be non-negative"):
        bitonic_topk(x, -1)

    with pytest.raises(ValueError, match="cannot exceed last dimension size"):
        bitonic_topk(x, 20)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_bitonic_topk_dtypes(dtype: torch.dtype) -> None:
    """Test bitonic top-k across diverse floating point dtypes."""
    x = torch.randn(8, 32, dtype=dtype)
    k = 8
    indices = bitonic_topk(x, k)
    ref = bitonic_topk_pytorch_ref(x, k)

    assert indices.shape == (8, 8)
    assert torch.equal(indices, ref)


def test_bitonic_topk_amp_cpu() -> None:
    """Verify torch.amp autocasting compliance on CPU."""
    x = torch.randn(4, 64)
    with torch.amp.autocast("cpu", dtype=torch.bfloat16):
        indices = bitonic_topk(x, 10)
    ref = bitonic_topk_pytorch_ref(x, 10)

    assert indices.shape == (4, 10)
    assert torch.equal(indices, ref)


def test_bitonic_topk_autograd_integration() -> None:
    """Verify autograd backward produces zero gradients for index operator without errors."""
    x = torch.randn(4, 16, requires_grad=True)
    indices = bitonic_topk(x, 4)

    # Convert indices to float and perform dummy scalar reduction
    loss = (indices.float() * x.gather(-1, indices)).sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_bitonic_topk_multidimensional() -> None:
    """Verify bitonic top-k on 3D and 4D tensors."""
    x = torch.randn(2, 3, 8, 32)
    k = 5
    indices = bitonic_topk(x, k)
    ref = bitonic_topk_pytorch_ref(x, k)

    assert indices.shape == (2, 3, 8, 5)
    assert torch.equal(indices, ref)


def test_bitonic_topk_force_pytorch_backend() -> None:
    """Verify explicit forcing of PyTorch backend."""
    x = torch.randn(2, 16)
    with force_kernel_backend("pytorch"):
        out = bitonic_topk(x, 4)
    ref = bitonic_topk_pytorch_ref(x, 4)
    assert torch.equal(out, ref)


@pytest.mark.gpu
def test_bitonic_topk_cuda_parity() -> None:
    """Test CUDA execution parity against PyTorch reference when CUDA is available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable on host")

    x = torch.randn(16, 128, device="cuda", dtype=torch.float32)
    k = 16
    with torch.amp.autocast("cuda", dtype=torch.float16):
        out = bitonic_topk(x, k)
    ref = bitonic_topk_pytorch_ref(x, k)

    assert torch.equal(out, ref)
