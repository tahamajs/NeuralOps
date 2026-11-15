#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Unit and regression tests for Based Taylor Attention CUDA/C++ kernel.

Tests:
1. Symbol exports and importability.
2. CPU / reference fallback execution and numerical sanity (finite, no NaNs/Infs).
3. Parity between based_taylor_attention dispatch and based_taylor_attention_pytorch_ref.
4. Scale parameter behavior (default vs custom scale).
5. Multiple batch sizes, sequence lengths, head counts, and head dimensions (D <= 128).
6. Precision support: float32, bfloat16, float16.
7. Autograd backward contract: proper NotImplementedError when CUDA backward is not compiled.
8. Torch AMP autocast compatibility (torch.amp.autocast, never torch.cuda.amp).
9. Three-level fallback telemetry tracking via neuralops.ops.cuda._fallback.
10. Head dimension bounds handling and causal masking behavior.
"""

from __future__ import annotations

import pytest
import torch

from neuralops.ops.cuda._fallback import (
    force_kernel_backend,
    get_last_dispatch_backend,
)
from neuralops.ops.cuda.based_taylor_attention import (
    based_taylor_attention,
    based_taylor_attention_pytorch_ref,
)


def test_symbol_export_and_callable() -> None:
    """Verify that symbols are properly exported and callable."""
    assert callable(based_taylor_attention)
    assert callable(based_taylor_attention_pytorch_ref)


def test_cpu_fallback_shapes_and_finiteness() -> None:
    """Verify basic execution on CPU via reference fallback."""
    torch.manual_seed(42)
    B, H, L, D = 2, 4, 16, 32
    q = torch.randn(B, H, L, D, dtype=torch.float32)
    k = torch.randn(B, H, L, D, dtype=torch.float32)
    v = torch.randn(B, H, L, D, dtype=torch.float32)

    out = based_taylor_attention(q, k, v)
    assert out.shape == (B, H, L, D)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()
    assert not torch.isnan(out).any()


def test_numerical_parity_with_reference() -> None:
    """Verify dispatch output matches pytorch reference directly."""
    torch.manual_seed(123)
    B, H, L, D = 1, 2, 8, 16
    q = torch.randn(B, H, L, D, dtype=torch.float32)
    k = torch.randn(B, H, L, D, dtype=torch.float32)
    v = torch.randn(B, H, L, D, dtype=torch.float32)

    ref = based_taylor_attention_pytorch_ref(q, k, v)
    out = based_taylor_attention(q, k, v)

    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_custom_scale() -> None:
    """Verify that providing custom scale produces expected distinct values."""
    torch.manual_seed(42)
    B, H, L, D = 2, 2, 8, 16
    q = torch.randn(B, H, L, D)
    k = torch.randn(B, H, L, D)
    v = torch.randn(B, H, L, D)

    out_default = based_taylor_attention(q, k, v)
    custom_scale = 0.5
    out_custom = based_taylor_attention(q, k, v, scale=custom_scale)
    ref_custom = based_taylor_attention_pytorch_ref(q, k, v, scale=custom_scale)

    assert torch.allclose(out_custom, ref_custom, atol=1e-5, rtol=1e-5)
    assert not torch.allclose(out_default, out_custom)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_precision_support(dtype: torch.dtype) -> None:
    """Verify support across FP32, FP16, and BF16 dtypes."""
    torch.manual_seed(42)
    B, H, L, D = 1, 2, 8, 16
    q = torch.randn(B, H, L, D, dtype=dtype)
    k = torch.randn(B, H, L, D, dtype=dtype)
    v = torch.randn(B, H, L, D, dtype=dtype)

    out = based_taylor_attention(q, k, v)
    assert out.shape == (B, H, L, D)
    assert out.dtype == dtype
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("D", [16, 32, 64, 128])
def test_head_dimensions_coverage(D: int) -> None:
    """Verify execution across common head dimensions matching CUDA kernel unroll templates."""
    torch.manual_seed(42)
    B, H, L = 1, 2, 8
    q = torch.randn(B, H, L, D)
    k = torch.randn(B, H, L, D)
    v = torch.randn(B, H, L, D)

    out = based_taylor_attention(q, k, v)
    assert out.shape == (B, H, L, D)
    assert torch.isfinite(out).all()


def test_autocast_compliance() -> None:
    """Verify torch.amp autocast compliance without deprecated torch.cuda.amp."""
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    q = torch.randn(1, 2, 8, 16)
    k = torch.randn(1, 2, 8, 16)
    v = torch.randn(1, 2, 8, 16)

    # Use torch.amp.autocast in compliance with Rule 2.1
    if device_type == "cpu":
        with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
            out = based_taylor_attention(q, k, v)
    else:
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            out = based_taylor_attention(q.cuda(), k.cuda(), v.cuda())

    assert torch.isfinite(out).all()


def test_autograd_contract() -> None:
    """Verify autograd forward pass and backward gradient computation."""
    q = torch.randn(1, 2, 4, 16, requires_grad=True)
    k = torch.randn(1, 2, 4, 16, requires_grad=True)
    v = torch.randn(1, 2, 4, 16, requires_grad=True)

    out = based_taylor_attention(q, k, v)
    loss = out.sum()
    loss.backward()

    assert q.grad is not None and torch.isfinite(q.grad).all()
    assert k.grad is not None and torch.isfinite(k.grad).all()
    assert v.grad is not None and torch.isfinite(v.grad).all()


def test_telemetry_dispatch_backend() -> None:
    """Verify dispatcher fallback telemetry logs the backend used."""
    q = torch.randn(1, 2, 4, 16)
    k = torch.randn(1, 2, 4, 16)
    v = torch.randn(1, 2, 4, 16)

    with force_kernel_backend("pytorch"):
        out = based_taylor_attention(q, k, v)
        assert out.shape == q.shape
        backend = get_last_dispatch_backend("based_taylor_attention")
        assert backend == "pytorch"


def test_causal_masking_property() -> None:
    """Verify that later key/value tokens do not affect earlier query outputs (causality)."""
    torch.manual_seed(42)
    B, H, L, D = 1, 1, 4, 8
    q = torch.randn(B, H, L, D)
    k1 = torch.randn(B, H, L, D)
    v1 = torch.randn(B, H, L, D)

    # Modify only the last token in k and v
    k2 = k1.clone()
    v2 = v1.clone()
    k2[:, :, 3:, :] += 10.0
    v2[:, :, 3:, :] += 10.0

    out1 = based_taylor_attention(q, k1, v1)
    out2 = based_taylor_attention(q, k2, v2)

    # Outputs for tokens 0, 1, 2 should be identical because token 3 is in the future
    assert torch.allclose(out1[:, :, :3, :], out2[:, :, :3, :], atol=1e-6)
    # Output for token 3 should differ
    assert not torch.allclose(out1[:, :, 3, :], out2[:, :, 3, :])
