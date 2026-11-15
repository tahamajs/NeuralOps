#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Unit and regression tests for Cross Attention CUDA kernel & CPU fallback (Wave 9 / W9-K19).

Tests:
1. Symbol exports, callables, and importability from neuralops.ops.cuda.cross_attention.
2. CPU / reference fallback execution and numerical sanity (finite, no NaNs/Infs).
3. Parity between cross_attention and pure PyTorch reference across asymmetric sequence lengths (L_q != L_kv).
4. Scale parameter behavior (default 1/sqrt(D) vs custom scale).
5. Multiple batch sizes, head counts, sequence lengths, and head dimensions (D <= 512).
6. Precision support: float32, bfloat16, float16.
7. Autograd backward pass behavior on CPU fallback.
8. Torch AMP autocast compliance (using torch.amp.autocast, Rule 2.1).
9. Top-level dispatcher integration via neuralops.ops.cuda._fallback.dispatch_cross_attention.
10. Fallback backend routing and forced backend support via force_kernel_backend.
"""

from __future__ import annotations

import math
import pytest
import torch
from torch.amp import autocast

from neuralops.ops.cuda._fallback import (
    dispatch_cross_attention,
    force_kernel_backend,
    is_kernel_available,
)
from neuralops.ops.cuda.cross_attention import (
    CrossAttentionFunction,
    cross_attention,
    cross_attention_forward,
    cross_attention_pytorch_ref,
)


def test_symbol_export_and_callables() -> None:
    """Verify that all public symbols are correctly exported and callable."""
    assert callable(cross_attention_forward)
    assert callable(cross_attention)
    assert callable(cross_attention_pytorch_ref)
    assert issubclass(CrossAttentionFunction, torch.autograd.Function)


def test_cpu_fallback_basic_shapes_and_finiteness() -> None:
    """Verify basic forward execution on CPU produces correct shapes and finite outputs."""
    torch.manual_seed(42)
    B, H, L_q, L_kv, D = 2, 4, 8, 16, 32
    q = torch.randn(B, H, L_q, D, dtype=torch.float32)
    k = torch.randn(B, H, L_kv, D, dtype=torch.float32)
    v = torch.randn(B, H, L_kv, D, dtype=torch.float32)

    out = cross_attention_forward(q, k, v)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (B, H, L_q, D)
    assert out.dtype == torch.float32
    assert out.device == q.device
    assert torch.isfinite(out).all()
    assert not torch.isnan(out).any()


def test_numerical_parity_with_reference() -> None:
    """Verify cross_attention output matches PyTorch reference across asymmetric lengths."""
    torch.manual_seed(123)
    B, H, L_q, L_kv, D = 2, 3, 7, 13, 64
    q = torch.randn(B, H, L_q, D, dtype=torch.float32)
    k = torch.randn(B, H, L_kv, D, dtype=torch.float32)
    v = torch.randn(B, H, L_kv, D, dtype=torch.float32)

    ref = cross_attention_pytorch_ref(q, k, v)
    out = cross_attention(q, k, v)

    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_custom_scale() -> None:
    """Verify custom scale parameter behavior."""
    torch.manual_seed(42)
    B, H, L_q, L_kv, D = 1, 2, 4, 8, 16
    q = torch.randn(B, H, L_q, D)
    k = torch.randn(B, H, L_kv, D)
    v = torch.randn(B, H, L_kv, D)

    out_default = cross_attention(q, k, v)
    custom_scale = 0.125
    out_custom = cross_attention(q, k, v, scale=custom_scale)
    ref_custom = cross_attention_pytorch_ref(q, k, v, scale=custom_scale)

    torch.testing.assert_close(out_custom, ref_custom, rtol=1e-5, atol=1e-5)
    assert not torch.allclose(out_default, out_custom)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_precision_support(dtype: torch.dtype) -> None:
    """Verify execution across FP32, BF16, and FP16 datatypes."""
    torch.manual_seed(42)
    B, H, L_q, L_kv, D = 1, 2, 4, 6, 32
    q = torch.randn(B, H, L_q, D, dtype=dtype)
    k = torch.randn(B, H, L_kv, D, dtype=dtype)
    v = torch.randn(B, H, L_kv, D, dtype=dtype)

    out = cross_attention(q, k, v)
    assert out.shape == (B, H, L_q, D)
    assert out.dtype == dtype
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("D", [16, 32, 64, 128, 256])
def test_head_dimensions_coverage(D: int) -> None:
    """Verify multiple head dimensions up to 256."""
    torch.manual_seed(42)
    B, H, L_q, L_kv = 1, 2, 4, 8
    q = torch.randn(B, H, L_q, D)
    k = torch.randn(B, H, L_kv, D)
    v = torch.randn(B, H, L_kv, D)

    out = cross_attention(q, k, v)
    assert out.shape == (B, H, L_q, D)
    assert torch.isfinite(out).all()


def test_autograd_cpu_fallback() -> None:
    """Verify autograd backward pass execution on CPU fallback."""
    q = torch.randn(2, 2, 4, 16, dtype=torch.float32, requires_grad=True)
    k = torch.randn(2, 2, 8, 16, dtype=torch.float32, requires_grad=True)
    v = torch.randn(2, 2, 8, 16, dtype=torch.float32, requires_grad=True)

    out = cross_attention_forward(q, k, v)
    loss = (out * 2.0).sum()
    loss.backward()

    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None
    assert q.grad.shape == q.shape
    assert k.grad.shape == k.shape
    assert v.grad.shape == v.shape
    assert torch.isfinite(q.grad).all()


def test_amp_autocast_compliance() -> None:
    """Verify torch.amp.autocast compliance (Rule 2.1)."""
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device_type == "cuda" else torch.bfloat16

    q = torch.randn(1, 2, 4, 16, device=device_type)
    k = torch.randn(1, 2, 8, 16, device=device_type)
    v = torch.randn(1, 2, 8, 16, device=device_type)

    with autocast(device_type=device_type, dtype=dtype):
        out = cross_attention(q, k, v)

    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, 2, 4, 16)
    assert torch.isfinite(out).all()


def test_dispatch_fallback_routing() -> None:
    """Verify dispatcher fallback routing with force_kernel_backend."""
    torch.manual_seed(42)
    B, H, L_q, L_kv, D = 1, 2, 4, 8, 16
    q = torch.randn(B, H, L_q, D)
    k = torch.randn(B, H, L_kv, D)
    v = torch.randn(B, H, L_kv, D)

    # Calling with pytorch backend forced through dispatcher
    with force_kernel_backend("pytorch"):
        out_py = dispatch_cross_attention(q, k, v, pytorch_fn=cross_attention_pytorch_ref)
        assert out_py.shape == (B, H, L_q, D)
        assert torch.isfinite(out_py).all()

    # Kernel availability check should return bool
    avail_cuda = is_kernel_available("cross_attention", backend="cuda")
    assert isinstance(avail_cuda, bool)
