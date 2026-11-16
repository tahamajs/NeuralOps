# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit and regression tests for Gated Linear Attention Scan CUDA kernel & CPU fallback (Wave 11 / W11-K11).

Tests:
1. Symbol exports, callability, and top-level integration in neuralops.ops.cuda.
2. CPU and reference fallback execution, tensor shapes, and numerical finiteness.
3. Numerical parity against gla_scan_pytorch_ref.
4. Autograd backward propagation for q, k, v, and optional alpha gating.
5. Partial gradient propagation when some tensors do not require gradients.
6. Direct GlaScanFunction.apply execution.
7. Strict temporal causality invariance.
8. Fallback backend routing and telemetry tracking.
9. AMP autocast compliance using torch.amp.autocast (never torch.cuda.amp).
10. Edge cases including sequence length zero, 4D alpha, and head_dim > 128.
11. Strict AGENTS.md rules compliance verification.
"""

from __future__ import annotations

import inspect
from pathlib import Path
import pytest
import torch
from torch.amp import autocast

import neuralops.ops.cuda as noc
from neuralops.ops.cuda._fallback import (
    force_kernel_backend,
    get_last_dispatch_backend,
)
from neuralops.ops.cuda.gla_scan import (
    GlaScanFunction,
    _Gla_scanFunction,
    gla_scan,
    gla_scan_pytorch_ref,
)


def test_symbol_exports_and_callability() -> None:
    """Verify that all public symbols are correctly exported and callable."""
    assert callable(gla_scan)
    assert callable(gla_scan_pytorch_ref)
    assert GlaScanFunction is _Gla_scanFunction
    assert issubclass(GlaScanFunction, torch.autograd.Function)

    # Verify export in neuralops.ops.cuda package
    assert hasattr(noc, "gla_scan")
    assert hasattr(noc, "gla_scan_pytorch_ref")
    assert callable(noc.gla_scan)
    assert callable(noc.gla_scan_pytorch_ref)

    # Verify inclusion in kernel catalog
    assert "gla_scan" in noc.list_cuda_kernels("operators")
    assert "gla_scan" in noc.CUDA_KERNEL_CATALOG


def test_cpu_fallback_basic_shapes_and_finiteness() -> None:
    """Verify execution on CPU fallback with finiteness and shape preservation."""
    torch.manual_seed(42)
    B, H, L, D = 2, 4, 16, 32
    q = torch.randn(B, H, L, D, dtype=torch.float32)
    k = torch.randn(B, H, L, D, dtype=torch.float32)
    v = torch.randn(B, H, L, D, dtype=torch.float32)
    alpha = torch.sigmoid(torch.randn(B, H, L, dtype=torch.float32))

    out = gla_scan(q, k, v, alpha=alpha)

    assert isinstance(out, torch.Tensor)
    assert out.shape == (B, H, L, D)
    assert out.dtype == torch.float32
    assert out.device == q.device
    assert torch.isfinite(out).all()
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_cpu_fallback_without_alpha() -> None:
    """Verify forward scan works properly when alpha is omitted (defaults to 1.0)."""
    torch.manual_seed(42)
    B, H, L, D = 1, 2, 8, 16
    q = torch.randn(B, H, L, D, dtype=torch.float32)
    k = torch.randn(B, H, L, D, dtype=torch.float32)
    v = torch.randn(B, H, L, D, dtype=torch.float32)

    out = gla_scan(q, k, v, alpha=None)

    assert out.shape == (B, H, L, D)
    assert torch.isfinite(out).all()
    assert not torch.isnan(out).any()


def test_numerical_parity_against_reference() -> None:
    """Verify numerical parity between dispatched operator and reference implementation."""
    torch.manual_seed(123)
    for D in (16, 32, 64):
        B, H, L = 2, 2, 12
        q = torch.randn(B, H, L, D, dtype=torch.float32)
        k = torch.randn(B, H, L, D, dtype=torch.float32)
        v = torch.randn(B, H, L, D, dtype=torch.float32)
        alpha = torch.sigmoid(torch.randn(B, H, L, dtype=torch.float32))

        out_op = gla_scan(q, k, v, alpha=alpha)
        out_ref = gla_scan_pytorch_ref(q, k, v, alpha=alpha)

        torch.testing.assert_close(out_op, out_ref, atol=1e-5, rtol=1e-5)


def test_top_level_noc_dispatch() -> None:
    """Verify neuralops.ops.cuda top-level access produces matching results."""
    torch.manual_seed(42)
    B, H, L, D = 1, 2, 8, 16
    q = torch.randn(B, H, L, D)
    k = torch.randn(B, H, L, D)
    v = torch.randn(B, H, L, D)
    alpha = torch.rand(B, H, L)

    out_direct = gla_scan(q, k, v, alpha)
    out_noc = noc.gla_scan(q, k, v, alpha)
    out_noc_ref = noc.gla_scan_pytorch_ref(q, k, v, alpha)

    torch.testing.assert_close(out_direct, out_noc)
    torch.testing.assert_close(out_noc, out_noc_ref)


def test_gla_scan_function_direct_apply() -> None:
    """Verify GlaScanFunction.apply executes directly on CPU."""
    B, H, L, D = 1, 2, 6, 16
    q = torch.randn(B, H, L, D, dtype=torch.float32)
    k = torch.randn(B, H, L, D, dtype=torch.float32)
    v = torch.randn(B, H, L, D, dtype=torch.float32)
    alpha = torch.rand(B, H, L, dtype=torch.float32)

    out = GlaScanFunction.apply(q, k, v, alpha)
    assert out.shape == (B, H, L, D)
    assert torch.isfinite(out).all()


def test_autograd_full_backward() -> None:
    """Verify analytical autograd gradients for all inputs (q, k, v, alpha)."""
    torch.manual_seed(42)
    B, H, L, D = 2, 2, 8, 16
    q = torch.randn(B, H, L, D, dtype=torch.float32, requires_grad=True)
    k = torch.randn(B, H, L, D, dtype=torch.float32, requires_grad=True)
    v = torch.randn(B, H, L, D, dtype=torch.float32, requires_grad=True)
    alpha = torch.sigmoid(torch.randn(B, H, L, dtype=torch.float32)).requires_grad_(True)

    out = gla_scan(q, k, v, alpha)
    loss = (out**2).sum()
    loss.backward()

    for name, tensor in [("q", q), ("k", k), ("v", v), ("alpha", alpha)]:
        grad = tensor.grad
        assert grad is not None, f"Gradient for {name} was not computed"
        assert grad.shape == tensor.shape, f"Gradient shape mismatch for {name}"
        assert torch.isfinite(grad).all(), f"Gradient for {name} contains non-finite values"
        assert not torch.isnan(grad).any(), f"Gradient for {name} contains NaNs"
        assert (grad != 0).any().item(), f"Gradient for {name} is all zeros"


def test_autograd_partial_requires_grad() -> None:
    """Verify autograd behaves cleanly when only a subset of inputs require gradients."""
    torch.manual_seed(42)
    B, H, L, D = 1, 2, 6, 8
    q = torch.randn(B, H, L, D, dtype=torch.float32, requires_grad=True)
    k = torch.randn(B, H, L, D, dtype=torch.float32, requires_grad=False)
    v = torch.randn(B, H, L, D, dtype=torch.float32, requires_grad=True)
    alpha = torch.rand(B, H, L, dtype=torch.float32, requires_grad=False)

    out = gla_scan(q, k, v, alpha)
    loss = out.sum()
    loss.backward()

    assert q.grad is not None and torch.isfinite(q.grad).all()
    assert k.grad is None
    assert v.grad is not None and torch.isfinite(v.grad).all()
    assert alpha.grad is None


def test_autograd_backward_without_alpha() -> None:
    """Verify autograd backward works as expected when alpha is None."""
    torch.manual_seed(42)
    B, H, L, D = 1, 2, 6, 8
    q = torch.randn(B, H, L, D, dtype=torch.float32, requires_grad=True)
    k = torch.randn(B, H, L, D, dtype=torch.float32, requires_grad=True)
    v = torch.randn(B, H, L, D, dtype=torch.float32, requires_grad=True)

    out = gla_scan(q, k, v, alpha=None)
    loss = out.sum()
    loss.backward()

    assert q.grad is not None and torch.isfinite(q.grad).all()
    assert k.grad is not None and torch.isfinite(k.grad).all()
    assert v.grad is not None and torch.isfinite(v.grad).all()


def test_strict_temporal_causality() -> None:
    """Verify that inputs at future time steps t' > t do not affect outputs at step t."""
    torch.manual_seed(42)
    B, H, L, D = 1, 2, 8, 16
    q1 = torch.randn(B, H, L, D)
    k1 = torch.randn(B, H, L, D)
    v1 = torch.randn(B, H, L, D)
    alpha1 = torch.rand(B, H, L)

    # Modify future tokens (time indices >= 4)
    q2 = q1.clone()
    k2 = k1.clone()
    v2 = v1.clone()
    alpha2 = alpha1.clone()

    k2[:, :, 4:, :] += 5.0
    v2[:, :, 4:, :] += 3.0
    q2[:, :, 4:, :] += 2.0
    alpha2[:, :, 4:] = 0.5

    out1 = gla_scan(q1, k1, v1, alpha1)
    out2 = gla_scan(q2, k2, v2, alpha2)

    # Outputs for tokens 0..3 must match exactly
    torch.testing.assert_close(out1[:, :, :4, :], out2[:, :, :4, :], atol=1e-6, rtol=1e-6)
    # Outputs for future tokens must differ
    assert not torch.allclose(out1[:, :, 4:, :], out2[:, :, 4:, :])


def test_amp_autocast_compliance() -> None:
    """Verify compatibility with torch.amp.autocast without torch.cuda.amp."""
    B, H, L, D = 1, 2, 8, 16
    q = torch.randn(B, H, L, D, dtype=torch.bfloat16)
    k = torch.randn(B, H, L, D, dtype=torch.bfloat16)
    v = torch.randn(B, H, L, D, dtype=torch.bfloat16)
    alpha = torch.rand(B, H, L, dtype=torch.bfloat16)

    with autocast("cpu", dtype=torch.bfloat16):
        out = gla_scan(q, k, v, alpha)

    assert out.shape == (B, H, L, D)
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out).all()


def test_fallback_backend_selection() -> None:
    """Verify backend forcing and dispatch telemetry tracking."""
    B, H, L, D = 1, 2, 4, 8
    q = torch.randn(B, H, L, D)
    k = torch.randn(B, H, L, D)
    v = torch.randn(B, H, L, D)

    with force_kernel_backend("pytorch"):
        out = gla_scan(q, k, v)
        backend = get_last_dispatch_backend("gla_scan")

    assert out.shape == (B, H, L, D)
    assert backend == "pytorch"


def test_edge_cases() -> None:
    """Verify edge cases: zero sequence length, 4D alpha, and large D (>128)."""
    # 1. Zero length sequence
    q0 = torch.randn(2, 2, 0, 16)
    k0 = torch.randn(2, 2, 0, 16)
    v0 = torch.randn(2, 2, 0, 16)
    out0 = gla_scan(q0, k0, v0)
    assert out0.shape == (2, 2, 0, 16)

    # 2. 4D alpha of shape (B, H, L, 1)
    q4 = torch.randn(1, 2, 4, 8)
    k4 = torch.randn(1, 2, 4, 8)
    v4 = torch.randn(1, 2, 4, 8)
    alpha_4d = torch.rand(1, 2, 4, 1)
    out4 = gla_scan(q4, k4, v4, alpha=alpha_4d)
    assert out4.shape == (1, 2, 4, 8)
    assert torch.isfinite(out4).all()

    # 3. D > 128 (dispatches cleanly to reference fallback)
    q_large = torch.randn(1, 2, 4, 256)
    k_large = torch.randn(1, 2, 4, 256)
    v_large = torch.randn(1, 2, 4, 256)
    out_large = gla_scan(q_large, k_large, v_large)
    assert out_large.shape == (1, 2, 4, 256)
    assert torch.isfinite(out_large).all()


def test_agents_md_rules_compliance() -> None:
    """Verify full compliance with AGENTS.md rules across the gla_scan module."""
    base_dir = Path("neuralops/ops/cuda/gla_scan")
    assert base_dir.is_dir()

    # Rule 6.3: Line count <= 800
    for file_path in base_dir.glob("*"):
        if file_path.is_file() and not file_path.name.endswith(".pyc"):
            text = file_path.read_text(encoding="utf-8")
            line_count = len(text.splitlines())
            assert line_count <= 800, f"{file_path.name} exceeds 800 lines limit ({line_count})"

    # Rule 2.1: No torch.cuda.amp
    # Rule 2.2: No time.time() for GPU timing
    # Rule 1.4 / 1.5: No main() entrypoint
    # Rule: No hardcoded cuda:0
    for py_file in base_dir.glob("*.py"):
        code = py_file.read_text(encoding="utf-8")
        assert "torch.cuda.amp" not in code, f"Deprecated torch.cuda.amp in {py_file.name}"
        assert "time.time()" not in code, f"time.time() timing pattern in {py_file.name}"
        assert "cuda:0" not in code, f"Hardcoded cuda:0 in {py_file.name}"
        assert 'if __name__ == "__main__":' not in code, f"Forbidden main block in {py_file.name}"
        assert "def main(" not in code, f"Forbidden main() in {py_file.name}"

    # Verify C++ / CUDA files
    for cu_file in list(base_dir.glob("*.cu")) + list(base_dir.glob("*.cpp")):
        src = cu_file.read_text(encoding="utf-8")
        assert "cuda:0" not in src, f"Hardcoded cuda:0 in {cu_file.name}"
