# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for Fused SVD Projection CUDA operator and PyTorch CPU fallbacks.
Verifies compliance with AGENTS.md rules, numerical sanity, and autograd.
"""

from __future__ import annotations

from pathlib import Path
import pytest
import torch
from torch.amp import autocast

import neuralops.ops.cuda as noc
from neuralops.ops.cuda._fallback import force_kernel_backend
from neuralops.ops.cuda.fused_svd_projection import (
    FusedSVDProjectionFunction,
    fused_svd_projection,
    fused_svd_projection_pytorch_ref,
)


class TestFusedSVDProjectionExports:
    """Verify symbol exports and callability across packages."""

    def test_top_level_cuda_export(self) -> None:
        """Verify fused_svd_projection is accessible via neuralops.ops.cuda."""
        assert hasattr(noc, "fused_svd_projection")
        assert callable(noc.fused_svd_projection)

    def test_subpackage_exports(self) -> None:
        """Verify exports in neuralops.ops.cuda.fused_svd_projection."""
        assert callable(fused_svd_projection)
        assert callable(fused_svd_projection_pytorch_ref)
        assert issubclass(FusedSVDProjectionFunction, torch.autograd.Function)


class TestFusedSVDProjectionExecution:
    """Verify forward and backward execution under CPU fallback."""

    def test_basic_fused_svd_projection_cpu(self) -> None:
        """Verify basic forward execution with all components on CPU."""
        M, K, R, N = 8, 32, 16, 24
        x = torch.randn(M, K, dtype=torch.float32)
        U = torch.randn(K, R, dtype=torch.float32)
        S = torch.randn(R, dtype=torch.float32)
        V_T = torch.randn(R, N, dtype=torch.float32)
        res = torch.randn(M, N, dtype=torch.float32)
        bias = torch.randn(N, dtype=torch.float32)

        out = fused_svd_projection(x, U, S=S, V_T=V_T, X_res=res, bias=bias, alpha=1.2, beta=0.8)

        assert out.shape == (M, N)
        assert out.dtype == torch.float32
        assert torch.isfinite(out).all()
        assert not torch.isnan(out).any()

    def test_top_level_dispatch(self) -> None:
        """Verify execution through noc.fused_svd_projection."""
        M, K, R, N = 4, 16, 8, 12
        x = torch.randn(M, K)
        U = torch.randn(K, R)
        S = torch.ones(R)
        V_T = torch.randn(R, N)

        out = noc.fused_svd_projection(x, U, S=S, V_T=V_T)
        assert out.shape == (M, N)
        assert torch.isfinite(out).all()

    def test_numerical_exact_parity(self) -> None:
        """Verify mathematical agreement: Y = alpha * ((X @ U) * S @ V_T) + beta * X_res + bias."""
        M, K, R, N = 6, 20, 10, 14
        alpha = 1.5
        beta = 0.5
        x = torch.randn(M, K, dtype=torch.float64)
        U = torch.randn(K, R, dtype=torch.float64)
        S = torch.randn(R, dtype=torch.float64)
        V_T = torch.randn(R, N, dtype=torch.float64)
        res = torch.randn(M, N, dtype=torch.float64)
        bias = torch.randn(N, dtype=torch.float64)

        out = fused_svd_projection(x, U, S=S, V_T=V_T, X_res=res, bias=bias, alpha=alpha, beta=beta)

        # Expected formula
        z = (x @ U) * S
        expected = alpha * (z @ V_T) + beta * res + bias

        assert torch.allclose(out, expected, atol=1e-6)

    def test_batch_3d_input(self) -> None:
        """Verify batched 3D inputs [Batch, SeqLen, K] are correctly projected."""
        B, L, K, R, N = 2, 7, 32, 8, 16
        x = torch.randn(B, L, K)
        U = torch.randn(K, R)
        S = torch.randn(R)
        V_T = torch.randn(R, N)
        res = torch.randn(B, L, N)
        bias = torch.randn(N)

        out = fused_svd_projection(x, U, S=S, V_T=V_T, X_res=res, bias=bias)
        assert out.shape == (B, L, N)
        assert torch.isfinite(out).all()

    def test_optional_components(self) -> None:
        """Verify execution when optional tensors are omitted."""
        M, K, R, N = 4, 16, 8, 12
        x = torch.randn(M, K)
        U = torch.randn(K, R)
        V_T = torch.randn(R, N)

        # Without S, X_res, bias
        out1 = fused_svd_projection(x, U, S=None, V_T=V_T, X_res=None, bias=None)
        assert out1.shape == (M, N)
        assert torch.isfinite(out1).all()

        # Without V_T (down-projection only)
        out2 = fused_svd_projection(x, U, S=None, V_T=None)
        assert out2.shape == (M, R)
        assert torch.isfinite(out2).all()

    def test_autograd_full_backward(self) -> None:
        """Verify analytic autograd backward pass across all inputs."""
        M, K, R, N = 4, 16, 8, 12
        x = torch.randn(M, K, requires_grad=True)
        U = torch.randn(K, R, requires_grad=True)
        S = torch.randn(R, requires_grad=True)
        V_T = torch.randn(R, N, requires_grad=True)
        res = torch.randn(M, N, requires_grad=True)
        bias = torch.randn(N, requires_grad=True)

        out = fused_svd_projection(x, U, S=S, V_T=V_T, X_res=res, bias=bias, alpha=1.2, beta=0.8)
        loss = out.sum()
        loss.backward()

        for tensor_name, grad, expected_shape in [
            ("x", x.grad, (M, K)),
            ("U", U.grad, (K, R)),
            ("S", S.grad, (R,)),
            ("V_T", V_T.grad, (R, N)),
            ("res", res.grad, (M, N)),
            ("bias", bias.grad, (N,)),
        ]:
            assert grad is not None, f"Gradient for {tensor_name} is None"
            assert grad.shape == expected_shape, f"Grad shape mismatch for {tensor_name}"
            assert torch.isfinite(grad).all(), f"Non-finite grad for {tensor_name}"
            assert not torch.isnan(grad).any(), f"NaN in grad for {tensor_name}"

    def test_autograd_partial_backward(self) -> None:
        """Verify backward pass when only a subset of inputs require gradients."""
        M, K, R, N = 4, 16, 8, 12
        x = torch.randn(M, K, requires_grad=True)
        U = torch.randn(K, R, requires_grad=False)
        S = torch.randn(R, requires_grad=False)
        V_T = torch.randn(R, N, requires_grad=True)

        out = fused_svd_projection(x, U, S=S, V_T=V_T)
        out.sum().backward()

        assert x.grad is not None and torch.isfinite(x.grad).all()
        assert U.grad is None
        assert S.grad is None
        assert V_T.grad is not None and torch.isfinite(V_T.grad).all()

    def test_forced_pytorch_dispatch(self) -> None:
        """Verify 3-level fallback dispatcher when forced to 'pytorch' backend."""
        M, K, R, N = 4, 16, 8, 12
        x = torch.randn(M, K)
        U = torch.randn(K, R)
        V_T = torch.randn(R, N)

        with force_kernel_backend("pytorch"):
            out = fused_svd_projection(x, U, V_T=V_T)
        assert out.shape == (M, N)
        assert torch.isfinite(out).all()

    def test_torch_amp_autocast_cpu(self) -> None:
        """Verify execution under torch.amp.autocast without errors."""
        M, K, R, N = 4, 16, 8, 12
        x = torch.randn(M, K, dtype=torch.float32)
        U = torch.randn(K, R, dtype=torch.float32)
        V_T = torch.randn(R, N, dtype=torch.float32)

        with autocast(device_type="cpu", dtype=torch.bfloat16):
            out = fused_svd_projection(x, U, V_T=V_T)
        assert out.shape == (M, N)
        assert torch.isfinite(out).all()


class TestFusedSVDProjectionAGENTSCompliance:
    """Verify code compliance with AGENTS.md rules."""

    def test_agents_compliance(self) -> None:
        """Verify rules 2.1, 2.2, 6.3, and no hardcoded cuda:0."""
        base_dir = (
            Path(__file__).resolve().parent.parent.parent.parent / "neuralops" / "ops" / "cuda" / "fused_svd_projection"
        )
        py_files = list(base_dir.glob("*.py"))
        cu_files = list(base_dir.glob("*.cu"))
        cpp_files = list(base_dir.glob("*.cpp"))

        all_files = py_files + cu_files + cpp_files
        assert len(all_files) >= 3, "Expected at least 3 source files in fused_svd_projection"

        for p in all_files:
            text = p.read_text(encoding="utf-8")
            lines = text.splitlines()

            # Rule 6.3: File size <= 800 lines
            assert len(lines) <= 800, f"{p.name} exceeds 800 lines ({len(lines)} lines)"

            # Rule 2.1: No deprecated torch.cuda.amp
            assert "torch.cuda.amp" not in text, f"{p.name} contains deprecated torch.cuda.amp"

            # Rule 2.2: No time.time() for GPU timing
            assert "time.time()" not in text, f"{p.name} contains time.time()"

            # No hardcoded cuda:0
            assert "cuda:0" not in text, f"{p.name} contains hardcoded cuda:0"
