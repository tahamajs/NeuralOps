# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for Activation-Aware SVD (ASVD) CUDA Kernel and PyTorch/CPU fallbacks.
"""

from __future__ import annotations

import pytest
import torch

from neuralops.ops.cuda._fallback import (
    dispatch_asvd,
    force_kernel_backend,
    is_kernel_available,
)
from neuralops.ops.cuda.asvd import ASVDFunction, asvd_compress
from neuralops.ops.triton.low_rank import asvd_compress_pytorch


class TestASVDKernelAudit:
    """Audit and functional verification test suite for ASVD operator."""

    def test_asvd_compress_shape_and_dtype(self) -> None:
        """Verify output shape and dtype preservation across float32 and float16."""
        for dtype in (torch.float32, torch.float16):
            W = torch.randn(64, 48, dtype=dtype)
            cov = torch.eye(48, dtype=dtype)
            out = asvd_compress(W, cov=cov, rank=16)
            assert out.shape == W.shape
            assert out.dtype == W.dtype
            assert not torch.isnan(out).any()
            assert not torch.isinf(out).any()

    def test_asvd_compress_no_covariance(self) -> None:
        """Verify behavior when cov is None (unweighted SVD with norm rebalancing)."""
        W = torch.randn(32, 32)
        out = asvd_compress(W, cov=None, rank=8)
        assert out.shape == W.shape
        assert not torch.isnan(out).any()

    def test_asvd_compress_invalid_dimensions(self) -> None:
        """Verify that non-2D tensors raise ValueError."""
        with pytest.raises(ValueError, match="requires a 2D weight matrix"):
            asvd_compress(torch.randn(16, 16, 16))

        with pytest.raises(ValueError, match="requires a 2D weight matrix"):
            asvd_compress(torch.randn(32))

    def test_asvd_compress_boundary_ranks(self) -> None:
        """Verify handling of rank boundary conditions (rank=1, rank >= min(M, N))."""
        W = torch.randn(30, 20)
        # rank = 1
        out_r1 = asvd_compress(W, rank=1)
        assert out_r1.shape == W.shape
        assert not torch.isnan(out_r1).any()

        # rank > min(M, N)
        out_large_r = asvd_compress(W, rank=100)
        assert out_large_r.shape == W.shape
        assert not torch.isnan(out_large_r).any()

        # 1x1 matrix edge case
        W_1x1 = torch.randn(1, 1)
        out_1x1 = asvd_compress(W_1x1, rank=1)
        assert out_1x1.shape == (1, 1)

    def test_asvd_autograd_backward_fallback(self) -> None:
        """Verify autograd gradient propagation through ASVDFunction reference fallback."""
        M, N, K = 16, 20, 8
        U = torch.randn(M, K, requires_grad=True)
        V = torch.randn(K, N, requires_grad=True)
        gain = torch.ones(M, requires_grad=True)

        out = ASVDFunction.apply(U, V, gain)
        loss = out.sum()
        loss.backward()

        assert U.grad is not None and U.grad.shape == U.shape
        assert V.grad is not None and V.grad.shape == V.shape
        assert gain.grad is not None and gain.grad.shape == gain.shape
        assert not torch.isnan(U.grad).any()
        assert not torch.isnan(V.grad).any()
        assert not torch.isnan(gain.grad).any()

    def test_asvd_autograd_without_gain(self) -> None:
        """Verify autograd gradient propagation when row_gain is None."""
        M, N, K = 12, 16, 6
        U = torch.randn(M, K, requires_grad=True)
        V = torch.randn(K, N, requires_grad=True)

        out = ASVDFunction.apply(U, V, None)
        loss = out.sum()
        loss.backward()

        assert U.grad is not None and U.grad.shape == U.shape
        assert V.grad is not None and V.grad.shape == V.shape

    def test_asvd_torch_amp_autocast_compliance(self) -> None:
        """Ensure full compatibility with torch.amp.autocast on CPU without error."""
        W = torch.randn(32, 32, dtype=torch.float32)
        with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
            out = asvd_compress(W, rank=16)
        assert out.shape == W.shape
        assert not torch.isnan(out).any()

    def test_asvd_three_level_dispatch_forced_pytorch(self) -> None:
        """Test 3-level fallback dispatcher when forced to 'pytorch' backend."""
        W = torch.randn(32, 32)
        with force_kernel_backend("pytorch"):
            out = dispatch_asvd(W, pytorch_fn=asvd_compress_pytorch, rank=16)
        assert out.shape == W.shape
        assert not torch.isnan(out).any()

    def test_asvd_three_level_dispatch_auto(self) -> None:
        """Test 3-level fallback dispatcher under auto mode."""
        W = torch.randn(32, 32)
        out = dispatch_asvd(W, pytorch_fn=asvd_compress_pytorch, rank=16)
        assert out.shape == W.shape
        assert not torch.isnan(out).any()

    def test_asvd_three_level_dispatch_forced_cuda_error(self) -> None:
        """Test that forcing 'cuda' on a non-CUDA host raises RuntimeError cleanly."""
        if not torch.cuda.is_available():
            W = torch.randn(16, 16)
            with pytest.raises(RuntimeError, match=r"\[asvd\] Backend forced to 'cuda'"):
                with force_kernel_backend("cuda"):
                    dispatch_asvd(W, pytorch_fn=asvd_compress_pytorch, rank=8)

    def test_asvd_numerical_parity_with_pytorch_ref(self) -> None:
        """Verify numerical consistency between asvd_compress and asvd_compress_pytorch."""
        torch.manual_seed(42)
        W = torch.randn(40, 40)
        cov = torch.eye(40) + 0.1 * torch.randn(40, 40)
        cov = cov @ cov.T  # symmetric positive semi-definite

        out_fn = asvd_compress(W, cov=cov, rank=16)
        out_ref = asvd_compress_pytorch(W, cov=cov, rank=16)

        diff = torch.norm(out_fn - out_ref).item()
        assert diff < 1e-4
