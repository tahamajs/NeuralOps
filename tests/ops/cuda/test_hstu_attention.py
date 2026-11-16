# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for HSTU Attention CUDA operator and reference implementation.

Verifies:
1. Symbol exports, callability, and catalog registration.
2. Direct and top-level lazy dispatcher execution.
3. Submodule shadowing prevention.
4. CPU / reference fallback correctness, numerical sanity (no NaNs, no Infs).
5. Numerical parity between dispatch and reference implementation.
6. Multi-dtype support (FP32, FP16, BF16).
7. Autograd gradient propagation across u, v, and decay.
8. Edge cases (zero-sized batch/sequence).
9. AGENTS.md compliance (no deprecated torch.cuda.amp, no time.time(), line limits, no hardcoded cuda:0).
"""

from __future__ import annotations

import importlib
from pathlib import Path
import pytest
import torch

import neuralops.ops.cuda as noc
from neuralops.ops.cuda.hstu_attention import (
    hstu_attention,
    hstu_attention_ref,
)


def test_hstu_attention_symbol_exports() -> None:
    """Verify primary and reference symbols are exported and callable."""
    assert hasattr(noc, "hstu_attention"), "noc.hstu_attention missing"
    assert hasattr(noc, "hstu_attention_ref"), "noc.hstu_attention_ref missing"
    assert callable(noc.hstu_attention), "noc.hstu_attention must be callable"
    assert callable(noc.hstu_attention_ref), "noc.hstu_attention_ref must be callable"

    # Catalog registration
    modules = noc.list_cuda_kernels(category="modules")
    assert "hstu_attention" in modules, "hstu_attention not in CUDA kernel catalog"


def test_hstu_attention_submodule_shadowing() -> None:
    """Importing submodule must not shadow top-level callable operator."""
    import neuralops.ops.cuda.hstu_attention.hstu_attention_autograd  # noqa: F401
    import neuralops.ops.cuda as noc_reloaded

    assert callable(noc_reloaded.hstu_attention), "noc.hstu_attention must remain callable"


def test_hstu_attention_cpu_basic() -> None:
    """Verify forward execution on CPU produces valid tensor with proper shape."""
    B, N, D = 2, 8, 16
    u = torch.randn(B, N, D, dtype=torch.float32)
    v = torch.randn(B, N, D, dtype=torch.float32)
    decay = torch.sigmoid(torch.randn(B, N, D, dtype=torch.float32))

    out = hstu_attention(u, v, decay)

    assert isinstance(out, torch.Tensor)
    assert out.shape == (B, N, D)
    assert out.dtype == torch.float32
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_hstu_attention_top_level_dispatch() -> None:
    """Verify top-level lazy dispatcher noc.hstu_attention dispatches cleanly."""
    B, N, D = 1, 4, 8
    u = torch.randn(B, N, D, dtype=torch.float32)
    v = torch.randn(B, N, D, dtype=torch.float32)
    decay = torch.sigmoid(torch.randn(B, N, D, dtype=torch.float32))

    out = noc.hstu_attention(u, v, decay)
    ref = noc.hstu_attention_ref(u, v, decay)

    assert out.shape == (B, N, D)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)
    assert not torch.isnan(out).any()


def test_hstu_attention_numerical_parity() -> None:
    """Verify forward numerical parity between operator and reference implementation."""
    torch.manual_seed(42)
    B, N, D = 3, 10, 32
    u = torch.randn(B, N, D, dtype=torch.float32)
    v = torch.randn(B, N, D, dtype=torch.float32)
    decay = torch.sigmoid(torch.randn(B, N, D, dtype=torch.float32))

    out = hstu_attention(u, v, decay, eps=1e-5)
    ref = hstu_attention_ref(u, v, decay, eps=1e-5)

    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_hstu_attention_dtypes(dtype: torch.dtype) -> None:
    """Verify operator supports FP32, FP16, and BF16 dtypes."""
    B, N, D = 2, 4, 8
    u = torch.randn(B, N, D, dtype=dtype)
    v = torch.randn(B, N, D, dtype=dtype)
    decay = torch.sigmoid(torch.randn(B, N, D, dtype=dtype))

    out = hstu_attention(u, v, decay)

    assert out.shape == (B, N, D)
    assert out.dtype == dtype
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_hstu_attention_autograd_gradients() -> None:
    """Verify backward gradient flow through u, v, and decay without NaNs."""
    torch.manual_seed(42)
    B, N, D = 2, 6, 12
    u = torch.randn(B, N, D, dtype=torch.float32, requires_grad=True)
    v = torch.randn(B, N, D, dtype=torch.float32, requires_grad=True)
    decay = torch.sigmoid(torch.randn(B, N, D, dtype=torch.float32)).requires_grad_(True)

    out = hstu_attention(u, v, decay)
    loss = (out**2).sum()
    loss.backward()

    assert u.grad is not None, "Gradient for u is missing"
    assert v.grad is not None, "Gradient for v is missing"
    assert decay.grad is not None, "Gradient for decay is missing"

    assert u.grad.shape == u.shape
    assert v.grad.shape == v.shape
    assert decay.grad.shape == decay.shape

    assert not torch.isnan(u.grad).any(), "NaN found in u.grad"
    assert not torch.isnan(v.grad).any(), "NaN found in v.grad"
    assert not torch.isnan(decay.grad).any(), "NaN found in decay.grad"


def test_hstu_attention_edge_cases() -> None:
    """Verify behavior on zero-sized tensors."""
    u_empty = torch.randn(0, 4, 8)
    v_empty = torch.randn(0, 4, 8)
    decay_empty = torch.randn(0, 4, 8)
    out_empty = hstu_attention(u_empty, v_empty, decay_empty)
    assert out_empty.shape == (0, 4, 8)

    u_seq0 = torch.randn(2, 0, 8)
    v_seq0 = torch.randn(2, 0, 8)
    decay_seq0 = torch.randn(2, 0, 8)
    out_seq0 = hstu_attention(u_seq0, v_seq0, decay_seq0)
    assert out_seq0.shape == (2, 0, 8)


def test_hstu_attention_compliance() -> None:
    """Verify AGENTS.md rules compliance for all files in module."""
    base = Path("neuralops/ops/cuda/hstu_attention")
    assert base.exists(), f"Path {base} does not exist"

    for file_path in base.glob("*"):
        if file_path.suffix in [".py", ".cu", ".cpp", ".cuh", ".h"]:
            text = file_path.read_text(encoding="utf-8")
            assert "torch.cuda.amp" not in text, f"Found torch.cuda.amp in {file_path}"
            assert "time.time()" not in text, f"Found time.time() in {file_path}"
            assert "cuda:0" not in text, f"Found hardcoded cuda:0 in {file_path}"
            assert len(text.splitlines()) <= 800, f"Line limit exceeded in {file_path}"
