#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Unit and regression tests for Gated DeltaNet CUDA kernel & fallback (Wave 11 / W11-K7).

Tests:
1. Symbol exports and importability (both direct module and lazy noc resolution).
2. CPU and reference fallback execution, shapes, and numerical sanity (no NaNs, no Infs).
3. Numerical parity between gated_deltanet dispatch and gated_deltanet_pytorch_ref.
4. Recurrent gate (g) parameter behavior (with and without decay gate).
5. Support for both 3D (B, H, L) and 4D (B, H, L, 1) gate tensors.
6. Precision support: float32, float16, bfloat16.
7. Torch AMP compliance using torch.amp.autocast (never torch.cuda.amp, Rule 2.1).
8. Autograd forward pass and backward gradient flow through q, k, v, beta, g.
9. Three-level fallback telemetry tracking via neuralops.ops.cuda._fallback.
10. Strict causal recurrent invariant: future inputs do not affect past outputs.
"""

from __future__ import annotations

import pytest
import torch
from torch.amp import autocast

import neuralops.ops.cuda as noc
from neuralops.ops.cuda._fallback import (
    force_kernel_backend,
    get_last_dispatch_backend,
)
from neuralops.ops.cuda.gated_deltanet import (
    gated_deltanet,
    gated_deltanet_pytorch_ref,
)


def test_symbol_export_and_callable() -> None:
    """Verify that gated_deltanet symbols are properly exported and callable."""
    assert callable(gated_deltanet)
    assert callable(gated_deltanet_pytorch_ref)

    # Verify lazy resolution via neuralops.ops.cuda
    assert callable(noc.gated_deltanet)
    assert callable(noc.gated_deltanet_pytorch_ref)
    assert "gated_deltanet" in noc.list_cuda_kernels(category="modules")


def test_cpu_fallback_shapes_and_finiteness() -> None:
    """Verify basic forward execution on CPU via reference fallback."""
    torch.manual_seed(42)
    B, H, L, D = 2, 4, 8, 16
    q = torch.randn(B, H, L, D, dtype=torch.float32)
    k = torch.randn(B, H, L, D, dtype=torch.float32)
    v = torch.randn(B, H, L, D, dtype=torch.float32)
    beta = torch.sigmoid(torch.randn(B, H, L, dtype=torch.float32))

    out = gated_deltanet(q, k, v, beta)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (B, H, L, D)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()
    assert not torch.isnan(out).any()


def test_numerical_parity_with_reference() -> None:
    """Verify dispatch output matches pytorch reference directly."""
    torch.manual_seed(123)
    B, H, L, D = 2, 2, 6, 12
    q = torch.randn(B, H, L, D, dtype=torch.float32)
    k = torch.randn(B, H, L, D, dtype=torch.float32)
    v = torch.randn(B, H, L, D, dtype=torch.float32)
    beta = torch.sigmoid(torch.randn(B, H, L, dtype=torch.float32))

    ref = gated_deltanet_pytorch_ref(q, k, v, beta)
    out = gated_deltanet(q, k, v, beta)

    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)

    # Parity check metrics (Rule 9.3)
    max_abs_diff = (out - ref).abs().max().item()
    assert max_abs_diff < 1e-5


def test_optional_decay_gate() -> None:
    """Verify behavior with and without the optional decay gate g."""
    torch.manual_seed(42)
    B, H, L, D = 2, 4, 8, 16
    q = torch.randn(B, H, L, D)
    k = torch.randn(B, H, L, D)
    v = torch.randn(B, H, L, D)
    beta = torch.sigmoid(torch.randn(B, H, L))
    g = torch.sigmoid(torch.randn(B, H, L))

    out_no_g = gated_deltanet(q, k, v, beta, g=None)
    out_with_g = gated_deltanet(q, k, v, beta, g=g)
    ref_with_g = gated_deltanet_pytorch_ref(q, k, v, beta, g=g)

    assert torch.isfinite(out_with_g).all()
    torch.testing.assert_close(out_with_g, ref_with_g, atol=1e-5, rtol=1e-5)
    # With nontrivial g decay, output must differ from g=None
    assert not torch.allclose(out_no_g, out_with_g)


@pytest.mark.parametrize("beta_4d", [False, True])
@pytest.mark.parametrize("g_4d", [False, True])
def test_gate_tensor_dimensions_3d_and_4d(beta_4d: bool, g_4d: bool) -> None:
    """Verify seamless handling of (B, H, L) and (B, H, L, 1) gate shapes."""
    torch.manual_seed(42)
    B, H, L, D = 2, 2, 4, 8
    q = torch.randn(B, H, L, D)
    k = torch.randn(B, H, L, D)
    v = torch.randn(B, H, L, D)

    beta_shape = (B, H, L, 1) if beta_4d else (B, H, L)
    g_shape = (B, H, L, 1) if g_4d else (B, H, L)
    beta = torch.sigmoid(torch.randn(*beta_shape))
    g = torch.sigmoid(torch.randn(*g_shape))

    out = gated_deltanet(q, k, v, beta, g=g)
    ref = gated_deltanet_pytorch_ref(q, k, v, beta, g=g)

    assert out.shape == (B, H, L, D)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_precision_support(dtype: torch.dtype) -> None:
    """Verify execution and type preservation across float32, float16, and bfloat16."""
    torch.manual_seed(42)
    B, H, L, D = 2, 2, 4, 8
    q = torch.randn(B, H, L, D, dtype=dtype)
    k = torch.randn(B, H, L, D, dtype=dtype)
    v = torch.randn(B, H, L, D, dtype=dtype)
    beta = torch.sigmoid(torch.randn(B, H, L, dtype=dtype))

    out = gated_deltanet(q, k, v, beta)
    assert out.shape == (B, H, L, D)
    assert out.dtype == dtype
    assert torch.isfinite(out).all()


def test_autocast_compliance() -> None:
    """Verify torch.amp compliance (Rule 2.1) without deprecated torch.cuda.amp."""
    torch.manual_seed(42)
    B, H, L, D = 2, 2, 4, 8
    q = torch.randn(B, H, L, D)
    k = torch.randn(B, H, L, D)
    v = torch.randn(B, H, L, D)
    beta = torch.randn(B, H, L)

    with autocast(device_type="cpu", dtype=torch.bfloat16):
        out = gated_deltanet(q, k, v, beta)

    assert out.shape == (B, H, L, D)
    assert torch.isfinite(out).all()


def test_autograd_all_inputs_backward() -> None:
    """Verify analytical gradient backpropagation through q, k, v, beta, and g."""
    torch.manual_seed(42)
    B, H, L, D = 2, 2, 4, 8
    q = torch.randn(B, H, L, D, requires_grad=True)
    k = torch.randn(B, H, L, D, requires_grad=True)
    v = torch.randn(B, H, L, D, requires_grad=True)
    beta = torch.randn(B, H, L, requires_grad=True)
    g = torch.randn(B, H, L, requires_grad=True)

    out = gated_deltanet(q, k, v, beta, g)
    loss = (out * 1.5).sum()
    loss.backward()

    for name, tensor in [("q", q), ("k", k), ("v", v), ("beta", beta), ("g", g)]:
        assert tensor.grad is not None, f"Gradient for {name} was not computed"
        assert tensor.grad.shape == tensor.shape, f"Gradient shape mismatch for {name}"
        assert torch.isfinite(tensor.grad).all(), f"Gradient for {name} contains non-finite values"


def test_autograd_partial_requires_grad() -> None:
    """Verify backward gradient flow when only a subset of inputs requires grad."""
    torch.manual_seed(42)
    B, H, L, D = 2, 2, 4, 8
    q = torch.randn(B, H, L, D, requires_grad=False)
    k = torch.randn(B, H, L, D, requires_grad=False)
    v = torch.randn(B, H, L, D, requires_grad=True)
    beta = torch.randn(B, H, L, requires_grad=False)

    out = gated_deltanet(q, k, v, beta)
    loss = out.sum()
    loss.backward()

    assert q.grad is None
    assert k.grad is None
    assert v.grad is not None
    assert beta.grad is None
    assert torch.isfinite(v.grad).all()


def test_telemetry_dispatch_backend() -> None:
    """Verify dispatcher fallback telemetry logs the backend executed."""
    q = torch.randn(1, 2, 4, 8)
    k = torch.randn(1, 2, 4, 8)
    v = torch.randn(1, 2, 4, 8)
    beta = torch.randn(1, 2, 4)

    with force_kernel_backend("pytorch"):
        out = gated_deltanet(q, k, v, beta)
        assert out.shape == q.shape
        backend = get_last_dispatch_backend("gated_deltanet")
        assert backend == "pytorch"


def test_causal_recurrence_invariant() -> None:
    """Verify strict causality: perturbing token at time t does not affect outputs at < t."""
    torch.manual_seed(42)
    B, H, L, D = 1, 2, 4, 8
    q = torch.randn(B, H, L, D)
    k1 = torch.randn(B, H, L, D)
    v1 = torch.randn(B, H, L, D)
    beta = torch.sigmoid(torch.randn(B, H, L))

    # Clone and modify only token t = 3 in k and v
    k2 = k1.clone()
    v2 = v1.clone()
    k2[:, :, 3:, :] += 5.0
    v2[:, :, 3:, :] += 5.0

    out1 = gated_deltanet(q, k1, v1, beta)
    out2 = gated_deltanet(q, k2, v2, beta)

    # Tokens 0, 1, 2 must produce identical outputs
    torch.testing.assert_close(out1[:, :, :3, :], out2[:, :, :3, :], atol=1e-6, rtol=1e-6)
    # Token 3 must differ due to causal update
    assert not torch.allclose(out1[:, :, 3, :], out2[:, :, 3, :])


@pytest.mark.parametrize("L", [1, 2, 7, 16])
@pytest.mark.parametrize("D", [8, 16, 32])
def test_various_sequence_lengths_and_dimensions(L: int, D: int) -> None:
    """Verify recurrence across single-step and various sequence lengths and head dims."""
    torch.manual_seed(42)
    B, H = 2, 2
    q = torch.randn(B, H, L, D)
    k = torch.randn(B, H, L, D)
    v = torch.randn(B, H, L, D)
    beta = torch.sigmoid(torch.randn(B, H, L))

    out = gated_deltanet(q, k, v, beta)
    ref = gated_deltanet_pytorch_ref(q, k, v, beta)

    assert out.shape == (B, H, L, D)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)
