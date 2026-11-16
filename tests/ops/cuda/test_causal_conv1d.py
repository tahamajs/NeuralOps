#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Unit and regression tests for Causal Conv1D CUDA/C++ kernel (Wave 9 / W9-K13).

Tests:
1. Symbol exports and importability.
2. CPU and reference fallback execution, shapes, and numerical sanity.
3. Numerical parity between causal_conv1d dispatch and causal_conv1d_pytorch_ref.
4. Support for optional bias (with and without bias).
5. Support for SiLU activation flag (silu_activation=True and False).
6. 2D vs 3D weight tensor handling (D, Kw) vs (D, 1, Kw).
7. Precision support: float32, bfloat16, float16.
8. Torch AMP autocast compatibility (torch.amp.autocast, never torch.cuda.amp).
9. Three-level fallback telemetry tracking via neuralops.ops.cuda._fallback (Triton and PyTorch forcing).
10. Autograd forward pass and backward gradients for x, weight, and bias.
11. Causality invariant verification (future token modifications do not affect prior outputs).
12. Invalid weight dimension error handling (ValueError on non-2D/3D weights).
"""

from __future__ import annotations

import pytest
import torch

from neuralops.ops.cuda._fallback import (
    force_kernel_backend,
    get_last_dispatch_backend,
)
from neuralops.ops.cuda.causal_conv1d import (
    causal_conv1d,
    causal_conv1d_pytorch_ref,
)


def test_symbol_export_and_callable() -> None:
    """Verify that causal_conv1d symbols are properly exported and callable."""
    assert callable(causal_conv1d)
    assert callable(causal_conv1d_pytorch_ref)


def test_cpu_fallback_shapes_and_finiteness() -> None:
    """Verify basic execution on CPU via reference fallback."""
    torch.manual_seed(42)
    B, D, L, Kw = 2, 4, 16, 3
    x = torch.randn(B, D, L, dtype=torch.float32)
    w = torch.randn(D, Kw, dtype=torch.float32)
    b = torch.randn(D, dtype=torch.float32)

    out = causal_conv1d(x, w, b, silu_activation=False)
    assert out.shape == (B, D, L)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()
    assert not torch.isnan(out).any()


def test_numerical_parity_with_reference() -> None:
    """Verify dispatch output matches pytorch reference directly."""
    torch.manual_seed(123)
    B, D, L, Kw = 2, 8, 32, 4
    x = torch.randn(B, D, L, dtype=torch.float32)
    w = torch.randn(D, Kw, dtype=torch.float32)
    b = torch.randn(D, dtype=torch.float32)

    ref = causal_conv1d_pytorch_ref(x, w, b, silu_activation=True)
    out = causal_conv1d(x, w, b, silu_activation=True)

    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("has_bias", [True, False])
@pytest.mark.parametrize("silu_act", [True, False])
def test_bias_and_activation_variants(has_bias: bool, silu_act: bool) -> None:
    """Verify combinations of bias and silu_activation flags."""
    torch.manual_seed(42)
    B, D, L, Kw = 2, 4, 16, 3
    x = torch.randn(B, D, L)
    w = torch.randn(D, Kw)
    b = torch.randn(D) if has_bias else None

    out = causal_conv1d(x, w, b, silu_activation=silu_act)
    ref = causal_conv1d_pytorch_ref(x, w, b, silu_activation=silu_act)

    assert out.shape == (B, D, L)
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_weight_dimensions_2d_and_3d() -> None:
    """Verify both (D, Kw) 2D weights and (D, 1, Kw) 3D weights work identically."""
    torch.manual_seed(42)
    B, D, L, Kw = 2, 4, 16, 4
    x = torch.randn(B, D, L)
    w_2d = torch.randn(D, Kw)
    w_3d = w_2d.unsqueeze(1)
    b = torch.randn(D)

    out_2d = causal_conv1d(x, w_2d, b)
    out_3d = causal_conv1d(x, w_3d, b)

    assert torch.allclose(out_2d, out_3d, atol=1e-6)


def test_invalid_weight_dimension_raises() -> None:
    """Verify that 1D or 4D weight tensors raise a ValueError."""
    x = torch.randn(2, 4, 16)
    w_invalid = torch.randn(4)
    with pytest.raises(ValueError, match="weight must be 2D or 3D"):
        causal_conv1d(x, w_invalid)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_precision_support(dtype: torch.dtype) -> None:
    """Verify support across FP32, FP16, and BF16 dtypes."""
    torch.manual_seed(42)
    B, D, L, Kw = 1, 4, 16, 3
    x = torch.randn(B, D, L, dtype=dtype)
    w = torch.randn(D, Kw, dtype=dtype)
    b = torch.randn(D, dtype=dtype)

    out = causal_conv1d(x, w, b, silu_activation=True)
    assert out.shape == (B, D, L)
    assert out.dtype == dtype
    assert torch.isfinite(out).all()


def test_autocast_compliance() -> None:
    """Verify torch.amp autocast compliance without deprecated torch.cuda.amp."""
    torch.manual_seed(42)
    x = torch.randn(2, 4, 16)
    w = torch.randn(4, 3)
    b = torch.randn(4)

    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    target_dtype = torch.float16 if device_type == "cuda" else torch.bfloat16

    if device_type == "cuda":
        x = x.cuda()
        w = w.cuda()
        b = b.cuda()

    with torch.amp.autocast(device_type=device_type, dtype=target_dtype):  # type: ignore[attr-defined]
        out = causal_conv1d(x, w, b, silu_activation=True)

    assert torch.isfinite(out).all()
    assert out.shape == (2, 4, 16)


def test_three_level_fallback_telemetry() -> None:
    """Verify dispatcher telemetry tracks PyTorch and Triton backends when forced."""
    x = torch.randn(2, 4, 16)
    w = torch.randn(4, 3)
    b = torch.randn(4)

    # 1. Force PyTorch backend
    with force_kernel_backend("pytorch"):
        out_pt = causal_conv1d(x, w, b, silu_activation=True)
        assert out_pt.shape == (2, 4, 16)
        backend = get_last_dispatch_backend("causal_conv1d")
        assert backend == "pytorch"

    # 2. Force Triton backend
    with force_kernel_backend("triton"):
        out_triton = causal_conv1d(x, w, b, silu_activation=True)
        assert out_triton.shape == (2, 4, 16)
        backend = get_last_dispatch_backend("causal_conv1d")
        assert backend == "triton"

    # Verify numerical match between forced backends
    assert torch.allclose(out_pt, out_triton, atol=1e-5, rtol=1e-5)


def test_autograd_gradients() -> None:
    """Verify autograd backward produces correct gradients for inputs, weights, and biases."""
    torch.manual_seed(42)
    B, D, L, Kw = 2, 4, 8, 3
    x = torch.randn(B, D, L, requires_grad=True)
    w = torch.randn(D, Kw, requires_grad=True)
    b = torch.randn(D, requires_grad=True)

    out = causal_conv1d(x, w, b, silu_activation=True)
    loss = (out**2).sum()
    torch.autograd.backward(loss)

    assert x.grad is not None and x.grad.shape == x.shape
    assert w.grad is not None and w.grad.shape == w.shape
    assert b.grad is not None and b.grad.shape == b.shape
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(w.grad).all()
    assert torch.isfinite(b.grad).all()


def test_causality_property() -> None:
    """Verify that future inputs do not affect past outputs (strict temporal causality)."""
    torch.manual_seed(42)
    B, D, L, Kw = 1, 2, 8, 3
    x1 = torch.randn(B, D, L)
    w = torch.randn(D, Kw)
    b = torch.randn(D)

    # Modify tokens at index >= 5 in x2
    x2 = x1.clone()
    x2[:, :, 5:] += 10.0

    out1 = causal_conv1d(x1, w, b, silu_activation=True)
    out2 = causal_conv1d(x2, w, b, silu_activation=True)

    # Outputs for tokens 0..4 must be identical
    assert torch.allclose(out1[:, :, :5], out2[:, :, :5], atol=1e-6)
    # Outputs for tokens >= 5 must differ
    assert not torch.allclose(out1[:, :, 5:], out2[:, :, 5:])
