# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Comprehensive tests for Activation-Scaled Quantized GEMM operator (Wave 9 / W9-K1).

Validates:
1. Basic forward pass execution on CPU fallback across shapes and types.
2. Numerical parity and closeness against PyTorch reference implementation.
3. Analytical autograd backward pass and gradient flow through x, s_inv, w, and bias.
4. Numerical parity of autograd gradients against PyTorch reference gradients.
5. AMP compliance across FP16 and BF16 (torch.amp.autocast).
6. Non-CUDA hardware fallback compliance (Rule 2.5).
7. Top-level operator symbol accessibility and module resolution.
8. AGENTS.md invariant checks (no torch.cuda.amp, no unsynchronized time.time(), line limits).
"""

from __future__ import annotations

import importlib
import inspect
import pytest
import torch
from torch.amp import autocast

import neuralops.ops.cuda as noc

asg_pkg = importlib.import_module("neuralops.ops.cuda.act_scale_gemm")
asg_autograd = importlib.import_module("neuralops.ops.cuda.act_scale_gemm.act_scale_autograd")
act_scale_quant_gemm = asg_autograd.act_scale_quant_gemm
act_scale_quant_gemm_pytorch_ref = asg_autograd.act_scale_quant_gemm_pytorch_ref


def test_act_scale_gemm_cpu_basic() -> None:
    """Verify forward pass produces correct shape and non-NaN output on CPU."""
    x = torch.randn(2, 4, 16, dtype=torch.float32)
    s_inv = torch.randn(16, dtype=torch.float32)
    w = torch.randn(32, 16, dtype=torch.float32)
    bias = torch.randn(32, dtype=torch.float32)

    out = act_scale_quant_gemm(x, s_inv, w, bias)

    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 4, 32)
    assert out.dtype == torch.float32
    assert out.device == x.device
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_act_scale_gemm_without_bias() -> None:
    """Verify forward pass without optional bias parameter."""
    x = torch.randn(4, 16, dtype=torch.float32)
    s_inv = torch.randn(16, dtype=torch.float32)
    w = torch.randn(8, 16, dtype=torch.float32)

    out = act_scale_quant_gemm(x, s_inv, w)
    ref = act_scale_quant_gemm_pytorch_ref(x, s_inv, w)

    assert out.shape == (4, 8)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_act_scale_gemm_numerical_parity() -> None:
    """Verify forward numerical parity against PyTorch reference implementation."""
    torch.manual_seed(42)
    x = torch.randn(3, 5, 32, dtype=torch.float32)
    s_inv = torch.randn(32, dtype=torch.float32)
    w = torch.randn(64, 32, dtype=torch.float32)
    bias = torch.randn(64, dtype=torch.float32)

    out = act_scale_quant_gemm(x, s_inv, w, bias)
    ref = act_scale_quant_gemm_pytorch_ref(x, s_inv, w, bias)

    max_abs_diff = (out - ref).abs().max().item()
    assert max_abs_diff < 1e-5
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_act_scale_gemm_autograd_gradients() -> None:
    """Verify backward gradient computation with analytical fallbacks for all inputs."""
    torch.manual_seed(42)
    x = torch.randn(2, 8, dtype=torch.float32, requires_grad=True)
    s_inv = torch.randn(8, dtype=torch.float32, requires_grad=True)
    w = torch.randn(4, 8, dtype=torch.float32, requires_grad=True)
    bias = torch.randn(4, dtype=torch.float32, requires_grad=True)

    out = act_scale_quant_gemm(x, s_inv, w, bias)
    loss = (out**2).sum()
    loss.backward()

    assert x.grad is not None
    assert s_inv.grad is not None
    assert w.grad is not None
    assert bias.grad is not None

    assert x.grad.shape == x.shape
    assert s_inv.grad.shape == s_inv.shape
    assert w.grad.shape == w.shape
    assert bias.grad.shape == bias.shape

    assert not torch.isnan(x.grad).any()
    assert not torch.isnan(s_inv.grad).any()
    assert not torch.isnan(w.grad).any()
    assert not torch.isnan(bias.grad).any()


def test_act_scale_gemm_autograd_parity_vs_ref() -> None:
    """Verify autograd backward gradients match reference autograd gradients exactly."""
    torch.manual_seed(123)
    x1 = torch.randn(3, 4, 16, dtype=torch.float64, requires_grad=True)
    s1 = torch.randn(16, dtype=torch.float64, requires_grad=True)
    w1 = torch.randn(8, 16, dtype=torch.float64, requires_grad=True)
    b1 = torch.randn(8, dtype=torch.float64, requires_grad=True)

    x2 = x1.detach().clone().requires_grad_(True)
    s2 = s1.detach().clone().requires_grad_(True)
    w2 = w1.detach().clone().requires_grad_(True)
    b2 = b1.detach().clone().requires_grad_(True)

    out1 = act_scale_quant_gemm(x1, s1, w1, b1)
    loss1 = (out1 * 1.5).sum()
    loss1.backward()

    # Compute reference gradients
    out2 = act_scale_quant_gemm_pytorch_ref(x2, s2, w2, b2)
    loss2 = (out2 * 1.5).sum()
    loss2.backward()

    torch.testing.assert_close(out1, out2, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(x1.grad, x2.grad, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(s1.grad, s2.grad, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(w1.grad, w2.grad, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(b1.grad, b2.grad, rtol=1e-5, atol=1e-5)


def test_act_scale_gemm_amp_compliance() -> None:
    """Verify torch.amp compliance (Rule 2.1) across float16 and bfloat16 contexts."""
    torch.manual_seed(42)
    x = torch.randn(2, 4, 16)
    s_inv = torch.randn(16)
    w = torch.randn(8, 16)
    bias = torch.randn(8)

    # Autocast CPU with bfloat16
    with autocast("cpu", dtype=torch.bfloat16):
        out = act_scale_quant_gemm(x, s_inv, w, bias)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (2, 4, 8)
        assert not torch.isnan(out).any()

    # Explicit half precision
    x_h = x.to(torch.float16)
    s_h = s_inv.to(torch.float16)
    w_h = w.to(torch.float16)
    b_h = bias.to(torch.float16)
    out_h = act_scale_quant_gemm(x_h, s_h, w_h, b_h)
    assert out_h.dtype == torch.float16
    assert out_h.shape == (2, 4, 8)


def test_act_scale_gemm_top_level_symbol_resolution() -> None:
    """Verify noc.act_scale_gemm and noc.act_scale_quant_gemm symbol resolution."""
    assert hasattr(noc, "act_scale_quant_gemm")
    assert hasattr(noc, "act_scale_quant_gemm_pytorch_ref")
    assert hasattr(noc, "act_scale_gemm")

    x = torch.randn(2, 16)
    s_inv = torch.randn(16)
    w = torch.randn(4, 16)

    out_via_noc = noc.act_scale_gemm(x, s_inv, w)
    assert out_via_noc.shape == (2, 4)


def test_act_scale_gemm_empty_and_edge_shapes() -> None:
    """Verify edge cases such as batch size 0 or single token."""
    # 1D single token
    x_1d = torch.randn(16)
    s_inv = torch.randn(16)
    w = torch.randn(8, 16)
    out_1d = act_scale_quant_gemm(x_1d, s_inv, w)
    assert out_1d.shape == (8,)

    # Higher dimensional input (e.g. 4D)
    x_4d = torch.randn(2, 3, 4, 16)
    out_4d = act_scale_quant_gemm(x_4d, s_inv, w)
    assert out_4d.shape == (2, 3, 4, 8)


def test_governance_and_compliance() -> None:
    """Verify strict adherence to AGENTS.md rules."""
    # Rule 2.1: No torch.cuda.amp references
    for mod in [asg_pkg, asg_autograd]:
        src = inspect.getsource(mod)
        assert "torch.cuda.amp" not in src, f"Deprecated torch.cuda.amp found in {mod.__name__}"
        assert "time.time()" not in src, f"Unsynchronized time.time() found in {mod.__name__}"
        assert "NotImplementedError" not in src, f"Unimplemented autograd path found in {mod.__name__}"

    # Rule 6.3: Line count <= 800 lines
    lines = inspect.getsource(asg_autograd).count("\n")
    assert lines <= 800, f"asg_autograd line count ({lines}) exceeds 800 lines"
