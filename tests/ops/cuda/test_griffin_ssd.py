# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for Griffin SSD Recurrent Operator and Reference Implementation.

Verifies:
1. Symbol exports, callability, and catalog registration.
2. Direct and top-level lazy dispatcher execution.
3. Submodule shadowing prevention.
4. CPU / reference fallback correctness, numerical sanity (no NaNs, no Infs).
5. Numerical parity between dispatch and reference implementation.
6. Multi-dtype support (FP32, FP16, BF16).
7. Autograd gradient propagation across x, r, and decay.
8. Edge cases (zero-sized batch/sequence/dim, non-contiguous inputs).
9. AGENTS.md compliance (no deprecated torch.cuda.amp, no time.time(), line limits, no hardcoded cuda:0).
"""

from __future__ import annotations

from pathlib import Path
import pytest
import torch

import neuralops.ops.cuda as noc
from neuralops.ops.cuda.griffin_ssd import (
    griffin_ssd,
    griffin_ssd_ref,
)


def test_griffin_ssd_symbol_exports() -> None:
    """Verify primary and reference symbols are exported and callable."""
    assert hasattr(noc, "griffin_ssd"), "noc.griffin_ssd missing"
    assert hasattr(noc, "griffin_ssd_ref"), "noc.griffin_ssd_ref missing"
    assert callable(noc.griffin_ssd), "noc.griffin_ssd must be callable"
    assert callable(noc.griffin_ssd_ref), "noc.griffin_ssd_ref must be callable"

    # Catalog registration
    modules = noc.list_cuda_kernels(category="modules")
    assert "griffin_ssd" in modules, "griffin_ssd not in CUDA kernel catalog modules"

    operators = noc.list_cuda_kernels(category="operators")
    assert "griffin_ssd" in operators, "griffin_ssd not in CUDA kernel catalog operators"


def test_griffin_ssd_submodule_shadowing() -> None:
    """Importing submodule must not shadow top-level callable operator."""
    import neuralops.ops.cuda.griffin_ssd.griffin_ssd_autograd  # noqa: F401
    import neuralops.ops.cuda as noc_reloaded

    assert callable(noc_reloaded.griffin_ssd), "noc.griffin_ssd must remain callable"


def test_griffin_ssd_cpu_basic() -> None:
    """Verify forward execution on CPU produces valid tensor with proper shape."""
    B, N, D = 2, 8, 16
    x = torch.randn(B, N, D, dtype=torch.float32)
    r = torch.randn(B, N, D, dtype=torch.float32)
    decay = torch.sigmoid(torch.randn(B, N, D, dtype=torch.float32))

    out = griffin_ssd(x, r, decay)

    assert isinstance(out, torch.Tensor)
    assert out.shape == (B, N, D)
    assert out.dtype == torch.float32
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_griffin_ssd_top_level_dispatch() -> None:
    """Verify top-level lazy dispatcher noc.griffin_ssd dispatches cleanly."""
    B, N, D = 1, 4, 8
    x = torch.randn(B, N, D, dtype=torch.float32)
    r = torch.randn(B, N, D, dtype=torch.float32)
    decay = torch.sigmoid(torch.randn(B, N, D, dtype=torch.float32))

    out = noc.griffin_ssd(x, r, decay)
    ref = noc.griffin_ssd_ref(x, r, decay)

    assert out.shape == (B, N, D)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_griffin_ssd_numerical_parity() -> None:
    """Verify forward numerical parity between operator and reference implementation."""
    torch.manual_seed(42)
    B, N, D = 3, 10, 32
    x = torch.randn(B, N, D, dtype=torch.float32)
    r = torch.randn(B, N, D, dtype=torch.float32)
    decay = torch.sigmoid(torch.randn(B, N, D, dtype=torch.float32))

    out = griffin_ssd(x, r, decay)
    ref = griffin_ssd_ref(x, r, decay)

    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_griffin_ssd_dtypes(dtype: torch.dtype) -> None:
    """Verify operator supports FP32, FP16, and BF16 dtypes."""
    B, N, D = 2, 4, 8
    x = torch.randn(B, N, D, dtype=dtype)
    r = torch.randn(B, N, D, dtype=dtype)
    decay = torch.sigmoid(torch.randn(B, N, D, dtype=dtype))

    out = griffin_ssd(x, r, decay)

    assert out.shape == (B, N, D)
    assert out.dtype == dtype
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_griffin_ssd_autograd_gradients() -> None:
    """Verify backward gradient flow through x, r, and decay without NaNs."""
    torch.manual_seed(42)
    B, N, D = 2, 6, 12
    x = torch.randn(B, N, D, dtype=torch.float32, requires_grad=True)
    r = torch.randn(B, N, D, dtype=torch.float32, requires_grad=True)
    decay = torch.sigmoid(torch.randn(B, N, D, dtype=torch.float32)).requires_grad_(True)

    out = griffin_ssd(x, r, decay)
    loss = (out**2).sum()
    loss.backward()

    assert x.grad is not None, "Gradient for x is missing"
    assert r.grad is not None, "Gradient for r is missing"
    assert decay.grad is not None, "Gradient for decay is missing"

    assert x.grad.shape == x.shape
    assert r.grad.shape == r.shape
    assert decay.grad.shape == decay.shape

    assert not torch.isnan(x.grad).any(), "NaN found in x.grad"
    assert not torch.isnan(r.grad).any(), "NaN found in r.grad"
    assert not torch.isnan(decay.grad).any(), "NaN found in decay.grad"


def test_griffin_ssd_edge_cases() -> None:
    """Verify behavior on zero-sized and non-contiguous tensors."""
    x_empty = torch.randn(0, 4, 8)
    r_empty = torch.randn(0, 4, 8)
    decay_empty = torch.randn(0, 4, 8)
    out_empty = griffin_ssd(x_empty, r_empty, decay_empty)
    assert out_empty.shape == (0, 4, 8)

    x_seq0 = torch.randn(2, 0, 8)
    r_seq0 = torch.randn(2, 0, 8)
    decay_seq0 = torch.randn(2, 0, 8)
    out_seq0 = griffin_ssd(x_seq0, r_seq0, decay_seq0)
    assert out_seq0.shape == (2, 0, 8)

    x_dim0 = torch.randn(2, 4, 0)
    r_dim0 = torch.randn(2, 4, 0)
    decay_dim0 = torch.randn(2, 4, 0)
    out_dim0 = griffin_ssd(x_dim0, r_dim0, decay_dim0)
    assert out_dim0.shape == (2, 4, 0)

    # Non-contiguous input
    x_nc = torch.randn(4, 2, 8).transpose(0, 1)
    r_nc = torch.randn(4, 2, 8).transpose(0, 1)
    decay_nc = torch.sigmoid(torch.randn(4, 2, 8).transpose(0, 1))
    assert not x_nc.is_contiguous()
    out_nc = griffin_ssd(x_nc, r_nc, decay_nc)
    assert out_nc.shape == (2, 4, 8)
    assert not torch.isnan(out_nc).any()


def test_griffin_ssd_compliance() -> None:
    """Verify AGENTS.md rules compliance for all files in griffin_ssd module."""
    base = Path("neuralops/ops/cuda/griffin_ssd")
    assert base.exists(), f"Path {base} does not exist"

    for file_path in base.glob("*"):
        if file_path.suffix in [".py", ".cu", ".cpp", ".cuh", ".h"]:
            text = file_path.read_text(encoding="utf-8")
            assert "torch.cuda.amp" not in text, f"Found torch.cuda.amp in {file_path}"
            assert "time.time()" not in text, f"Found time.time() in {file_path}"
            assert "cuda:0" not in text, f"Found hardcoded cuda:0 in {file_path}"
            assert len(text.splitlines()) <= 800, f"Line limit exceeded in {file_path}"
