# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for HNC-SVD CUDA Kernel and PyTorch/CPU fallbacks.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import neuralops.ops.cuda as noc
from neuralops.ops.cuda._fallback import (
    dispatch_hnc_svd,
    force_kernel_backend,
    is_kernel_available,
)
from neuralops.ops.cuda.hnc_svd import HNCSVDFunction, hnc_svd, hnc_svd_compress
from neuralops.ops.triton.low_rank import hnc_svd_compress_pytorch


class TestHNCSVDKernelAudit:
    """Audit and functional verification test suite for HNC-SVD operator."""

    def test_symbol_exports_and_callability(self) -> None:
        """Verify symbol exports and callability from module and top-level package."""
        assert callable(hnc_svd)
        assert callable(hnc_svd_compress)
        assert callable(HNCSVDFunction.apply)
        assert callable(noc.hnc_svd)
        assert callable(noc.hnc_svd_compress)
        assert "hnc_svd" in noc.list_cuda_kernels("modules")

    def test_hnc_svd_compress_shape_and_dtype(self) -> None:
        """Verify output shape and dtype preservation across float32, float16, and bfloat16."""
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            W = torch.randn(64, 48, dtype=dtype)
            cov = torch.eye(48, dtype=dtype)
            out = hnc_svd_compress(W, cov=cov, rank=16)
            assert out.shape == W.shape
            assert out.dtype == W.dtype
            assert not torch.isnan(out).any()
            assert not torch.isinf(out).any()

    def test_hnc_svd_compress_no_covariance(self) -> None:
        """Verify behavior when cov is None (unwhitened SVD compression)."""
        W = torch.randn(32, 32)
        out = hnc_svd_compress(W, cov=None, rank=8)
        assert out.shape == W.shape
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_hnc_svd_top_level_dispatch(self) -> None:
        """Verify top-level package alias noc.hnc_svd works identically."""
        torch.manual_seed(42)
        W = torch.randn(32, 32)
        out_alias = noc.hnc_svd(W, rank=8)
        out_func = noc.hnc_svd_compress(W, rank=8)
        assert out_alias.shape == W.shape
        assert torch.allclose(out_alias, out_func)

    def test_hnc_svd_compress_invalid_dimensions(self) -> None:
        """Verify that non-2D tensors raise ValueError."""
        with pytest.raises(ValueError, match="requires a 2D weight matrix"):
            hnc_svd_compress(torch.randn(16, 16, 16))

        with pytest.raises(ValueError, match="requires a 2D weight matrix"):
            hnc_svd_compress(torch.randn(32))

    def test_hnc_svd_compress_boundary_ranks(self) -> None:
        """Verify handling of rank boundary conditions (rank=1, rank >= min(M, N))."""
        # Rectangular matrix M > N
        W_tall = torch.randn(40, 20)
        out_r1 = hnc_svd_compress(W_tall, rank=1)
        assert out_r1.shape == W_tall.shape
        assert not torch.isnan(out_r1).any()

        # Rectangular matrix M < N
        W_wide = torch.randn(20, 40)
        out_wide = hnc_svd_compress(W_wide, rank=8)
        assert out_wide.shape == W_wide.shape
        assert not torch.isnan(out_wide).any()

        # rank > min(M, N)
        out_large_r = hnc_svd_compress(W_tall, rank=100)
        assert out_large_r.shape == W_tall.shape
        assert not torch.isnan(out_large_r).any()

        # 1x1 matrix edge case
        W_1x1 = torch.randn(1, 1)
        out_1x1 = hnc_svd_compress(W_1x1, rank=1)
        assert out_1x1.shape == (1, 1)
        assert not torch.isnan(out_1x1).any()

    def test_hnc_svd_autograd_backward_fallback(self) -> None:
        """Verify autograd gradient propagation through HNCSVDFunction reference fallback."""
        M, N, K = 16, 20, 8
        A = torch.randn(M, K, requires_grad=True)
        B = torch.randn(K, N, requires_grad=True)

        out = HNCSVDFunction.apply(A, B)
        loss = out.sum()
        loss.backward()

        assert A.grad is not None and A.grad.shape == A.shape
        assert B.grad is not None and B.grad.shape == B.shape
        assert not torch.isnan(A.grad).any()
        assert not torch.isnan(B.grad).any()
        assert not torch.isinf(A.grad).any()
        assert not torch.isinf(B.grad).any()

    def test_hnc_svd_torch_amp_autocast_compliance(self) -> None:
        """Ensure full compatibility with torch.amp.autocast on CPU without error."""
        W = torch.randn(32, 32, dtype=torch.float32)
        with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
            out = hnc_svd_compress(W, rank=16)
        assert out.shape == W.shape
        assert not torch.isnan(out).any()

    def test_hnc_svd_three_level_dispatch_forced_pytorch(self) -> None:
        """Test 3-level fallback dispatcher when forced to 'pytorch' backend."""
        W = torch.randn(32, 32)
        with force_kernel_backend("pytorch"):
            out = dispatch_hnc_svd(W, pytorch_fn=hnc_svd_compress_pytorch, rank=16)
        assert out.shape == W.shape
        assert not torch.isnan(out).any()

    def test_hnc_svd_three_level_dispatch_auto(self) -> None:
        """Test 3-level fallback dispatcher under auto mode."""
        W = torch.randn(32, 32)
        out = dispatch_hnc_svd(W, pytorch_fn=hnc_svd_compress_pytorch, rank=16)
        assert out.shape == W.shape
        assert not torch.isnan(out).any()

    def test_hnc_svd_three_level_dispatch_forced_cuda_error(self) -> None:
        """Test that forcing 'cuda' on a non-CUDA host raises RuntimeError cleanly."""
        if not torch.cuda.is_available():
            W = torch.randn(16, 16)
            with pytest.raises(RuntimeError, match=r"\[hnc_svd\] Backend forced to 'cuda'"):
                with force_kernel_backend("cuda"):
                    dispatch_hnc_svd(W, pytorch_fn=hnc_svd_compress_pytorch, rank=8)

    def test_hnc_svd_numerical_parity_with_pytorch_ref(self) -> None:
        """Verify numerical parity between hnc_svd_compress and hnc_svd_compress_pytorch."""
        torch.manual_seed(42)
        W = torch.randn(48, 48)
        cov = torch.eye(48) + 0.05 * torch.randn(48, 48)
        cov = cov @ cov.T  # symmetric positive semi-definite

        # With covariance
        out_fn = hnc_svd_compress(W, cov=cov, rank=16)
        out_ref = hnc_svd_compress_pytorch(W, cov=cov, rank=16)
        diff = torch.norm(out_fn - out_ref).item()
        assert diff < 1e-4

        # Without covariance
        out_fn_nocov = hnc_svd_compress(W, cov=None, rank=16)
        out_ref_nocov = hnc_svd_compress_pytorch(W, cov=None, rank=16)
        diff_nocov = torch.norm(out_fn_nocov - out_ref_nocov).item()
        assert diff_nocov < 1e-4

    def test_hnc_svd_compliance(self) -> None:
        """Verify compliance with AGENTS.md rules for hnc_svd."""
        base = Path("neuralops/ops/cuda/hnc_svd")
        for p in base.glob("*"):
            if p.suffix in (".py", ".cpp", ".cu", ".cuh"):
                text = p.read_text(encoding="utf-8")
                # Rule 2.1: No deprecated torch.cuda.amp
                assert "torch.cuda.amp" not in text, f"Deprecated torch.cuda.amp in {p}"
                # Rule 2.2: No time.time() for GPU timing
                assert "time.time()" not in text, f"Forbidden time.time() in {p}"
                # Rule 6.3: Line count <= 800
                lines = text.splitlines()
                assert len(lines) <= 800, f"File {p} exceeds 800 lines ({len(lines)})"
                # No hardcoded cuda:0
                assert "cuda:0" not in text, f"Hardcoded cuda:0 found in {p}"
