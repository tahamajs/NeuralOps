# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for Fast Walsh-Hadamard Transform (FWHT) CUDA operator and reference fallback (Wave 11 / W11-K4).

Verifies:
1. Symbol exports, callability, and catalog registration in neuralops.ops.cuda.
2. CPU and reference fallback execution, tensor shape/device/dtype preservation.
3. Numerical sanity (finite outputs, no NaNs, no Infs).
4. Exact numerical parity against PyTorch reference implementation.
5. Mathematical invariants of normalized FWHT (involutory self-inverse, Parseval energy conservation).
6. Execution across 1D, 2D, 3D, and 4D tensor layouts and various power-of-2 block dimensions.
7. Non-contiguous (strided) input tensor handling.
8. Precision support across FP32, FP64, FP16, and BF16, plus torch.amp autocast compliance.
9. Analytical autograd backward gradient flow and gradient parity.
10. Input validation error handling for non-power-of-2 and non-positive block dimensions.
11. Backend forcing and fallback behavior.
12. Strict AGENTS.md rule compliance (line limits, no torch.cuda.amp, no unsynchronized time.time(),
    no hardcoded cuda:0, no main entrypoint in library).
"""

from __future__ import annotations

import importlib
import types
from pathlib import Path

import pytest
import torch
from torch.amp import autocast

import neuralops.ops.cuda as noc
from neuralops.ops.cuda._fallback import force_kernel_backend
from neuralops.ops.cuda.fwht_transform import (
    fast_hadamard_transform,
    fwht_transform,
    fwht_transform_pytorch_ref,
)


def test_fwht_transform_symbol_exports_and_callability() -> None:
    """Verify symbol export, callability, catalog presence, and shadowing prevention."""
    # Direct module imports
    assert callable(fwht_transform)
    assert callable(fwht_transform_pytorch_ref)
    assert callable(fast_hadamard_transform)

    # Top-level lazy dispatcher imports
    assert hasattr(noc, "fwht_transform")
    assert callable(noc.fwht_transform)
    assert hasattr(noc, "fwht_transform_pytorch_ref")
    assert callable(noc.fwht_transform_pytorch_ref)
    assert hasattr(noc, "fast_hadamard_transform")
    assert callable(noc.fast_hadamard_transform)

    # Catalog registration
    assert "fwht_transform" in noc.CUDA_KERNEL_CATALOG
    assert "fwht_transform" in noc.list_cuda_kernels("modules")
    assert "fwht_transform" in noc.list_cuda_kernels("operators")

    # Submodule shadowing prevention
    submodule = importlib.import_module("neuralops.ops.cuda.fwht_transform")
    assert isinstance(submodule, types.ModuleType)
    resolved_attr = noc.fwht_transform
    assert callable(resolved_attr)
    assert not isinstance(resolved_attr, types.ModuleType)


def test_fwht_transform_cpu_fallback_basic() -> None:
    """Verify forward pass on CPU fallback preserves shape, dtype, device and produces finite values."""
    x = torch.randn(4, 128, dtype=torch.float32)
    out = noc.fwht_transform(x, B_dim=128)

    assert isinstance(out, torch.Tensor)
    assert out.shape == x.shape
    assert out.dtype == x.dtype
    assert out.device == x.device
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_fwht_transform_numerical_parity_vs_pytorch_ref() -> None:
    """Verify forward output matches PyTorch reference implementation."""
    torch.manual_seed(42)
    for b_dim in (32, 64, 128):
        x = torch.randn(3, b_dim * 2, dtype=torch.float32)
        out = noc.fwht_transform(x, B_dim=b_dim)
        ref = noc.fwht_transform_pytorch_ref(x, B_dim=b_dim)

        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


def test_fwht_transform_involutory_property() -> None:
    """Verify normalized FWHT is an involution: applying it twice recovers the original tensor."""
    torch.manual_seed(123)
    x = torch.randn(4, 64, dtype=torch.float32)

    # First transform
    y = noc.fwht_transform(x, B_dim=64)
    # Second transform (involutory property: H * H = I)
    recovered = noc.fwht_transform(y, B_dim=64)

    torch.testing.assert_close(recovered, x, rtol=1e-5, atol=1e-5)


def test_fwht_transform_energy_conservation_parseval() -> None:
    """Verify Parseval's theorem: orthogonal FWHT preserves Frobenius/L2 norm."""
    torch.manual_seed(999)
    x = torch.randn(8, 128, dtype=torch.float32)
    y = noc.fwht_transform(x, B_dim=128)

    orig_norm = torch.linalg.norm(x)
    transformed_norm = torch.linalg.norm(y)

    torch.testing.assert_close(transformed_norm, orig_norm, rtol=1e-5, atol=1e-5)


def test_fwht_transform_various_dimensions_and_blocks() -> None:
    """Verify FWHT across 1D, 2D, 3D, and 4D layouts and various power-of-2 block sizes."""
    torch.manual_seed(77)

    # 1D: (D,)
    x1d = torch.randn(128, dtype=torch.float32)
    out1d = noc.fwht_transform(x1d, B_dim=64)
    ref1d = noc.fwht_transform_pytorch_ref(x1d, B_dim=64)
    assert out1d.shape == x1d.shape
    torch.testing.assert_close(out1d, ref1d, rtol=1e-5, atol=1e-5)

    # 2D: (B, D)
    x2d = torch.randn(4, 64, dtype=torch.float32)
    out2d = noc.fwht_transform(x2d, B_dim=32)
    ref2d = noc.fwht_transform_pytorch_ref(x2d, B_dim=32)
    assert out2d.shape == x2d.shape
    torch.testing.assert_close(out2d, ref2d, rtol=1e-5, atol=1e-5)

    # 3D: (B, S, D)
    x3d = torch.randn(2, 4, 32, dtype=torch.float32)
    out3d = noc.fwht_transform(x3d, B_dim=16)
    ref3d = noc.fwht_transform_pytorch_ref(x3d, B_dim=16)
    assert out3d.shape == x3d.shape
    torch.testing.assert_close(out3d, ref3d, rtol=1e-5, atol=1e-5)

    # 4D: (B, H, S, D)
    x4d = torch.randn(2, 2, 4, 16, dtype=torch.float32)
    out4d = noc.fwht_transform(x4d, B_dim=8)
    ref4d = noc.fwht_transform_pytorch_ref(x4d, B_dim=8)
    assert out4d.shape == x4d.shape
    torch.testing.assert_close(out4d, ref4d, rtol=1e-5, atol=1e-5)


def test_fwht_transform_non_contiguous_input() -> None:
    """Verify non-contiguous (strided) inputs execute properly without failure."""
    raw = torch.randn(4, 256, dtype=torch.float32)
    x_strided = raw[:, ::2]  # shape: (4, 128), non-contiguous
    assert not x_strided.is_contiguous()

    out = noc.fwht_transform(x_strided, B_dim=64)
    ref = noc.fwht_transform_pytorch_ref(x_strided, B_dim=64)

    assert out.shape == x_strided.shape
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_fwht_transform_dtypes_and_amp() -> None:
    """Verify execution across supported dtypes and compliance with torch.amp (Rule 2.1)."""
    # Float32
    x_fp32 = torch.randn(2, 64, dtype=torch.float32)
    out_fp32 = noc.fwht_transform(x_fp32, B_dim=64)
    assert out_fp32.dtype == torch.float32

    # Float64
    x_fp64 = torch.randn(2, 64, dtype=torch.float64)
    out_fp64 = noc.fwht_transform(x_fp64, B_dim=64)
    assert out_fp64.dtype == torch.float64

    # Float16
    x_fp16 = torch.randn(2, 64, dtype=torch.float16)
    out_fp16 = noc.fwht_transform(x_fp16, B_dim=64)
    assert out_fp16.dtype == torch.float16

    # BFloat16
    x_bf16 = torch.randn(2, 64, dtype=torch.bfloat16)
    out_bf16 = noc.fwht_transform(x_bf16, B_dim=64)
    assert out_bf16.dtype == torch.bfloat16

    # torch.amp autocast compliance
    with autocast("cpu", dtype=torch.bfloat16):
        x_ctx = torch.randn(2, 64)
        out_ctx = noc.fwht_transform(x_ctx, B_dim=64)
        assert isinstance(out_ctx, torch.Tensor)
        assert not torch.isnan(out_ctx).any()


def test_fwht_transform_autograd_backward() -> None:
    """Verify backward gradient computation and gradient parity against reference."""
    torch.manual_seed(42)
    x = torch.randn(2, 64, dtype=torch.float32, requires_grad=True)
    out = noc.fwht_transform(x, B_dim=64)
    loss = (out**2).sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert not torch.isnan(x.grad).any()
    assert not torch.isinf(x.grad).any()

    # Verify gradient parity against pure PyTorch reference
    torch.manual_seed(42)
    x_ref = torch.randn(2, 64, dtype=torch.float32, requires_grad=True)
    ref_out = noc.fwht_transform_pytorch_ref(x_ref, B_dim=64)
    ref_loss = (ref_out**2).sum()
    ref_loss.backward()

    torch.testing.assert_close(x.grad, x_ref.grad, rtol=1e-5, atol=1e-5)


def test_fwht_transform_validation_errors() -> None:
    """Verify appropriate validation errors for invalid block dimension configurations."""
    x = torch.randn(2, 64)

    # Non-power-of-2 B_dim
    for invalid_b in (3, 5, 15, 100):
        with pytest.raises(ValueError, match="must be a positive power of 2"):
            noc.fwht_transform(x, B_dim=invalid_b)

        with pytest.raises(ValueError, match="must be a positive power of 2"):
            noc.fwht_transform_pytorch_ref(x, B_dim=invalid_b)

    # Non-positive B_dim (zero or negative)
    for non_positive_b in (0, -1, -64):
        with pytest.raises(ValueError, match="must be a positive power of 2"):
            noc.fwht_transform(x, B_dim=non_positive_b)

        with pytest.raises(ValueError, match="must be a positive power of 2"):
            noc.fwht_transform_pytorch_ref(x, B_dim=non_positive_b)


def test_fwht_transform_forced_backends() -> None:
    """Verify behavior under forced dispatch backends (Rule 2.5)."""
    x = torch.randn(2, 64, dtype=torch.float32)

    # Forced PyTorch backend
    with force_kernel_backend("pytorch"):
        out_py = noc.fwht_transform(x, B_dim=64)
        assert out_py.shape == x.shape
        assert not torch.isnan(out_py).any()

    # Forced CUDA backend on CPU tensor: fails cleanly or falls back
    if not torch.cuda.is_available():
        with force_kernel_backend("cuda"):
            with pytest.raises((RuntimeError, ValueError)):
                noc.fwht_transform(x, B_dim=64)


def test_fwht_transform_agents_md_compliance() -> None:
    """Verify codebase adherence to AGENTS.md rules for fwht_transform."""
    kernel_dir = Path("neuralops/ops/cuda/fwht_transform")
    assert kernel_dir.is_dir()

    # Check all files in directory
    for file_path in kernel_dir.glob("*"):
        if file_path.is_file() and file_path.suffix in (".py", ".cu", ".cpp", ".cuh"):
            content = file_path.read_text(encoding="utf-8")
            lines = content.splitlines()

            # Rule 6.3: File size <= 800 lines
            assert len(lines) <= 800, f"{file_path.name} exceeds 800 lines ({len(lines)})"

            # Rule 2.1: No torch.cuda.amp
            assert "torch.cuda.amp" not in content, f"{file_path.name} contains deprecated torch.cuda.amp"

            # Rule 2.2: No time.time() for GPU timing
            assert "time.time()" not in content, f"{file_path.name} contains time.time()"

            # No hardcoded cuda:0
            assert "cuda:0" not in content, f"{file_path.name} contains hardcoded cuda:0"

            # Rule 1.4 & Rule 1.5: No main() entrypoint in library
            if file_path.suffix == ".py":
                assert '__name__ == "__main__"' not in content, f"{file_path.name} contains main entrypoint"
