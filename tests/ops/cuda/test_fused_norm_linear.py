# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit tests for Fused Normalization Linear Projection Operator (Wave 11 / W11-K1).

Validates:
1. Top-level symbol exports and callability via neuralops.ops.cuda.
2. CPU and reference fallback execution correctness across dimensions (2D, 3D, 4D).
3. Forward numerical parity against PyTorch reference implementation.
4. Autograd backward gradient computation and parity against PyTorch reference.
5. AMP compliance (torch.amp.autocast, Rule 2.1).
6. Non-CUDA hardware device fallback and dispatch telemetry (Rule 2.5).
7. AGENTS.md governance compliance (line limits, no torch.cuda.amp, no time.time()).
"""

from __future__ import annotations

import importlib
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

fnl_pkg = importlib.import_module("neuralops.ops.cuda.fused_norm_linear")
fnl_autograd = importlib.import_module("neuralops.ops.cuda.fused_norm_linear.norm_linear_autograd")
fused_norm_linear = fnl_autograd.fused_norm_linear
fused_norm_linear_pytorch_ref = fnl_autograd.fused_norm_linear_pytorch_ref


def test_fused_norm_linear_symbol_exports_and_callability() -> None:
    """Verify top-level symbol accessibility and callability from neuralops.ops.cuda."""
    assert hasattr(noc, "fused_norm_linear"), "noc must export fused_norm_linear"
    assert hasattr(noc, "fused_norm_linear_pytorch_ref"), "noc must export fused_norm_linear_pytorch_ref"
    assert callable(noc.fused_norm_linear), "noc.fused_norm_linear must be callable"
    assert callable(noc.fused_norm_linear_pytorch_ref), "noc.fused_norm_linear_pytorch_ref must be callable"

    assert "fused_norm_linear" in fnl_pkg.__all__
    assert "fused_norm_linear_pytorch_ref" in fnl_pkg.__all__


def test_fused_norm_linear_cpu_basic() -> None:
    """Verify forward execution on CPU returns finite, non-NaN results of correct shape."""
    x = torch.randn(2, 4, 16, dtype=torch.float32)
    weight = torch.randn(32, 16, dtype=torch.float32)
    bias = torch.randn(32, dtype=torch.float32)

    out = fused_norm_linear(x, weight, bias=bias)

    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 4, 32)
    assert out.dtype == torch.float32
    assert out.device == x.device
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_fused_norm_linear_without_bias() -> None:
    """Verify forward execution without optional bias parameter."""
    x = torch.randn(4, 16, dtype=torch.float32)
    weight = torch.randn(8, 16, dtype=torch.float32)

    out = fused_norm_linear(x, weight, bias=None)
    ref = fused_norm_linear_pytorch_ref(x, weight, bias=None)

    assert out.shape == (4, 8)
    assert not torch.isnan(out).any()
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_fused_norm_linear_numerical_parity() -> None:
    """Verify exact numerical consistency with PyTorch reference implementation."""
    torch.manual_seed(42)
    x = torch.randn(3, 5, 32, dtype=torch.float32)
    weight = torch.randn(64, 32, dtype=torch.float32)
    bias = torch.randn(64, dtype=torch.float32)

    out = fused_norm_linear(x, weight, bias=bias, eps=1e-5)
    ref = fused_norm_linear_pytorch_ref(x, weight, bias=bias, eps=1e-5)

    max_abs_diff = (out - ref).abs().max().item()
    assert max_abs_diff < 1e-5
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_fused_norm_linear_autograd_gradients() -> None:
    """Verify autograd backward propagation and gradient finiteness for all parameters."""
    torch.manual_seed(42)
    x = torch.randn(2, 8, dtype=torch.float32, requires_grad=True)
    weight = torch.randn(4, 8, dtype=torch.float32, requires_grad=True)
    bias = torch.randn(4, dtype=torch.float32, requires_grad=True)

    out = fused_norm_linear(x, weight, bias=bias)
    loss = (out**2).sum()
    loss.backward()

    assert x.grad is not None
    assert weight.grad is not None
    assert bias.grad is not None

    assert x.grad.shape == x.shape
    assert weight.grad.shape == weight.shape
    assert bias.grad.shape == bias.shape

    assert not torch.isnan(x.grad).any()
    assert not torch.isnan(weight.grad).any()
    assert not torch.isnan(bias.grad).any()
    assert not torch.isinf(x.grad).any()
    assert not torch.isinf(weight.grad).any()
    assert not torch.isinf(bias.grad).any()


def test_fused_norm_linear_autograd_gradient_parity() -> None:
    """Verify backward gradient parity against PyTorch reference implementation."""
    torch.manual_seed(123)
    x1 = torch.randn(3, 4, 16, dtype=torch.float64, requires_grad=True)
    w1 = torch.randn(8, 16, dtype=torch.float64, requires_grad=True)
    b1 = torch.randn(8, dtype=torch.float64, requires_grad=True)

    x2 = x1.detach().clone().requires_grad_(True)
    w2 = w1.detach().clone().requires_grad_(True)
    b2 = b1.detach().clone().requires_grad_(True)

    out1 = fused_norm_linear(x1, w1, bias=b1, eps=1e-5)
    loss1 = (out1 * 2.0).sum()
    loss1.backward()

    out2 = fused_norm_linear_pytorch_ref(x2, w2, bias=b2, eps=1e-5)
    loss2 = (out2 * 2.0).sum()
    loss2.backward()

    torch.testing.assert_close(out1, out2, rtol=1e-5, atol=1e-5)
    assert x1.grad is not None and x2.grad is not None
    assert w1.grad is not None and w2.grad is not None
    assert b1.grad is not None and b2.grad is not None

    torch.testing.assert_close(x1.grad, x2.grad, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(w1.grad, w2.grad, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(b1.grad, b2.grad, rtol=1e-5, atol=1e-5)


def test_fused_norm_linear_edge_cases_and_shapes() -> None:
    """Verify various input tensor ranks (2D, 3D, 4D) and boundary dimensions."""
    # 2D input
    x_2d = torch.randn(6, 16)
    w = torch.randn(12, 16)
    b = torch.randn(12)
    out_2d = fused_norm_linear(x_2d, w, b)
    assert out_2d.shape == (6, 12)

    # 4D input
    x_4d = torch.randn(2, 3, 4, 16)
    out_4d = fused_norm_linear(x_4d, w, b)
    assert out_4d.shape == (2, 3, 4, 12)

    # Single token (1D batch)
    x_1t = torch.randn(1, 16)
    out_1t = fused_norm_linear(x_1t, w, b)
    assert out_1t.shape == (1, 12)

    # noc entrypoint parity
    out_noc = noc.fused_norm_linear(x_2d, w, b)
    torch.testing.assert_close(out_2d, out_noc)


def test_fused_norm_linear_amp_compliance() -> None:
    """Verify torch.amp compliance (Rule 2.1) across float16 and bfloat16 contexts."""
    torch.manual_seed(42)
    x = torch.randn(2, 4, 16)
    w = torch.randn(8, 16)
    b = torch.randn(8)

    # Autocast CPU with bfloat16
    with autocast("cpu", dtype=torch.bfloat16):
        out = fused_norm_linear(x, w, b)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (2, 4, 8)
        assert not torch.isnan(out).any()

    # Explicit half precision
    x_h = x.to(torch.float16)
    w_h = w.to(torch.float16)
    b_h = b.to(torch.float16)
    out_h = fused_norm_linear(x_h, w_h, b_h)
    assert out_h.dtype == torch.float16
    assert out_h.shape == (2, 4, 8)


def test_fused_norm_linear_dispatch_routing_and_telemetry() -> None:
    """Verify fallback routing and telemetry tracking under forced backends."""
    x = torch.randn(2, 4, 16)
    w = torch.randn(8, 16)
    b = torch.randn(8)

    # 1. Force PyTorch backend
    with force_kernel_backend("pytorch"):
        out_pt = fused_norm_linear(x, w, b)
        assert out_pt.shape == (2, 4, 8)
        backend = get_last_dispatch_backend("fused_norm_linear")
        assert backend == "pytorch"

    # 2. Force CUDA backend on non-CUDA host raises RuntimeError cleanly (Rule 2.5)
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match=r"\[fused_norm_linear\] Backend forced to 'cuda'"):
            with force_kernel_backend("cuda"):
                fused_norm_linear(x, w, b)


def test_fused_norm_linear_agents_md_compliance() -> None:
    """Verify strict adherence to AGENTS.md rules across fused_norm_linear module files."""
    module_dir = Path("neuralops/ops/cuda/fused_norm_linear")
    assert module_dir.exists() and module_dir.is_dir()

    for file_path in module_dir.iterdir():
        if file_path.suffix in (".py", ".cu", ".cpp"):
            content = file_path.read_text(encoding="utf-8")
            line_count = len(content.splitlines())

            # Rule 6.3: File size <= 800 lines
            assert line_count <= 800, f"{file_path.name} line count ({line_count}) exceeds hard limit of 800 lines"

            # Rule 2.1: AMP API - no torch.cuda.amp
            assert "torch.cuda.amp" not in content, f"Deprecated torch.cuda.amp found in {file_path.name}"

            # Rule 2.2: CUDA Timing - no time.time() for GPU timing
            assert "time.time()" not in content, f"Unsynchronized time.time() found in {file_path.name}"

            # Rule 10: No hardcoded cuda:0
            assert "cuda:0" not in content, f"Hardcoded cuda:0 found in {file_path.name}"

            # Rule 10: No silent except Exception: pass
            lines = content.splitlines()
            for i, line in enumerate(lines):
                if "except Exception:" in line and i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    assert next_line != "pass", f"Bare except Exception: pass in {file_path.name} line {i+1}"
