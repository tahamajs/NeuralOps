#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Unit and regression tests for GLA Chunk Attention CUDA/C++ kernel and PyTorch fallback.

Tests:
1. Symbol exports and callability (local module and top-level neuralops.ops.cuda).
2. CPU reference fallback execution and numerical sanity (finite, no NaNs/Infs).
3. Numerical parity between gla_chunk_attn dispatch and gla_chunk_attn_pytorch_ref.
4. Support for decay gating matrix (L, L) and feature-space gating (L, D).
5. Custom scale parameter behavior vs default 1/sqrt(D).
6. Precision support: float32, float16, bfloat16.
7. Diverse sequence lengths, batch sizes, and head dimensions.
8. Causal masking invariant (future keys/values do not affect past queries).
9. Autograd gradient flow and backward differentiability.
10. Torch AMP autocast compatibility (torch.amp.autocast, never torch.cuda.amp).
11. Three-level fallback telemetry tracking via neuralops.ops.cuda._fallback.
12. Static compliance with AGENTS.md rules (line count, no torch.cuda.amp, no time.time).
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

import neuralops.ops.cuda as noc
from neuralops.ops.cuda._fallback import (
    force_kernel_backend,
    get_last_dispatch_backend,
)
from neuralops.ops.cuda.gla_chunk_attn import (
    gla_chunk_attn,
    gla_chunk_attn_pytorch_ref,
)


def test_symbol_export_and_callable() -> None:
    """Verify that symbols are properly exported and callable from both package and top-level."""
    assert callable(gla_chunk_attn)
    assert callable(gla_chunk_attn_pytorch_ref)
    assert callable(noc.gla_chunk_attn)
    assert callable(noc.gla_chunk_attn_pytorch_ref)


def test_cpu_fallback_shapes_and_finiteness() -> None:
    """Verify basic execution on CPU via reference fallback."""
    torch.manual_seed(42)
    B, H, L, D = 2, 4, 16, 32
    q = torch.randn(B, H, L, D, dtype=torch.float32)
    k = torch.randn(B, H, L, D, dtype=torch.float32)
    v = torch.randn(B, H, L, D, dtype=torch.float32)

    out = gla_chunk_attn(q, k, v)
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

    ref = gla_chunk_attn_pytorch_ref(q, k, v)
    out = gla_chunk_attn(q, k, v)

    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_gated_attention_modes() -> None:
    """Verify execution with decay gating matrix (L, L) and feature gating (L, D)."""
    torch.manual_seed(42)
    B, H, L, D = 2, 2, 8, 16
    q = torch.randn(B, H, L, D)
    k = torch.randn(B, H, L, D)
    v = torch.randn(B, H, L, D)

    # Mode 1: Intra-chunk decay mask (B, H, L, L)
    g_decay = torch.sigmoid(torch.randn(B, H, L, L))
    out_decay = gla_chunk_attn(q, k, v, g=g_decay)
    ref_decay = gla_chunk_attn_pytorch_ref(q, k, v, g=g_decay)
    assert out_decay.shape == (B, H, L, D)
    assert torch.isfinite(out_decay).all()
    assert torch.allclose(out_decay, ref_decay, atol=1e-5, rtol=1e-5)

    # Mode 2: Per-channel feature gate (B, H, L, D)
    g_feat = torch.sigmoid(torch.randn(B, H, L, D))
    out_feat = gla_chunk_attn(q, k, v, g=g_feat)
    ref_feat = gla_chunk_attn_pytorch_ref(q, k, v, g=g_feat)
    assert out_feat.shape == (B, H, L, D)
    assert torch.isfinite(out_feat).all()
    assert torch.allclose(out_feat, ref_feat, atol=1e-5, rtol=1e-5)


def test_custom_scale() -> None:
    """Verify that providing custom scale produces expected distinct values."""
    torch.manual_seed(42)
    B, H, L, D = 2, 2, 8, 16
    q = torch.randn(B, H, L, D)
    k = torch.randn(B, H, L, D)
    v = torch.randn(B, H, L, D)

    out_default = gla_chunk_attn(q, k, v)
    custom_scale = 0.5
    out_custom = gla_chunk_attn(q, k, v, scale=custom_scale)
    ref_custom = gla_chunk_attn_pytorch_ref(q, k, v, scale=custom_scale)

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

    out = gla_chunk_attn(q, k, v)
    assert out.shape == (B, H, L, D)
    assert out.dtype == dtype
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("D", [16, 32, 64, 128])
def test_head_dimensions_coverage(D: int) -> None:
    """Verify execution across various head dimensions."""
    torch.manual_seed(42)
    B, H, L = 1, 2, 8
    q = torch.randn(B, H, L, D)
    k = torch.randn(B, H, L, D)
    v = torch.randn(B, H, L, D)

    out = gla_chunk_attn(q, k, v)
    assert out.shape == (B, H, L, D)
    assert torch.isfinite(out).all()


def test_causal_masking_property() -> None:
    """Verify that later key/value tokens do not affect earlier query outputs."""
    torch.manual_seed(42)
    B, H, L, D = 1, 1, 4, 8
    q = torch.randn(B, H, L, D)
    k1 = torch.randn(B, H, L, D)
    v1 = torch.randn(B, H, L, D)

    # Modify future token (index 3)
    k2 = k1.clone()
    v2 = v1.clone()
    k2[:, :, 3:, :] += 10.0
    v2[:, :, 3:, :] += 10.0

    out1 = gla_chunk_attn(q, k1, v1)
    out2 = gla_chunk_attn(q, k2, v2)

    # Outputs for tokens 0, 1, 2 should be identical because token 3 is in the future
    assert torch.allclose(out1[:, :, :3, :], out2[:, :, :3, :], atol=1e-6)
    # Output for token 3 should differ
    assert not torch.allclose(out1[:, :, 3, :], out2[:, :, 3, :])


def test_autograd_contract() -> None:
    """Verify autograd forward pass and backward gradient computation."""
    torch.manual_seed(42)
    B, H, L, D = 1, 2, 4, 16

    # Test without gate
    q = torch.randn(B, H, L, D, requires_grad=True)
    k = torch.randn(B, H, L, D, requires_grad=True)
    v = torch.randn(B, H, L, D, requires_grad=True)

    out = gla_chunk_attn(q, k, v)
    loss = out.sum()
    loss.backward()

    assert q.grad is not None and torch.isfinite(q.grad).all()
    assert k.grad is not None and torch.isfinite(k.grad).all()
    assert v.grad is not None and torch.isfinite(v.grad).all()

    # Test with decay gate
    q2 = torch.randn(B, H, L, D, requires_grad=True)
    k2 = torch.randn(B, H, L, D, requires_grad=True)
    v2 = torch.randn(B, H, L, D, requires_grad=True)
    g2 = torch.randn(B, H, L, L, requires_grad=True)

    out2 = gla_chunk_attn(q2, k2, v2, g2)
    loss2 = out2.sum()
    loss2.backward()

    assert q2.grad is not None and torch.isfinite(q2.grad).all()
    assert k2.grad is not None and torch.isfinite(k2.grad).all()
    assert v2.grad is not None and torch.isfinite(v2.grad).all()
    assert g2.grad is not None and torch.isfinite(g2.grad).all()

    # Test with feature gate
    q3 = torch.randn(B, H, L, D, requires_grad=True)
    k3 = torch.randn(B, H, L, D, requires_grad=True)
    v3 = torch.randn(B, H, L, D, requires_grad=True)
    g3 = torch.randn(B, H, L, D, requires_grad=True)

    out3 = gla_chunk_attn(q3, k3, v3, g3)
    loss3 = out3.sum()
    loss3.backward()

    assert q3.grad is not None and torch.isfinite(q3.grad).all()
    assert k3.grad is not None and torch.isfinite(k3.grad).all()
    assert v3.grad is not None and torch.isfinite(v3.grad).all()
    assert g3.grad is not None and torch.isfinite(g3.grad).all()


def test_autocast_compliance() -> None:
    """Verify torch.amp autocast compliance without deprecated torch.cuda.amp."""
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    q = torch.randn(1, 2, 8, 16)
    k = torch.randn(1, 2, 8, 16)
    v = torch.randn(1, 2, 8, 16)

    # Use torch.amp.autocast in compliance with Rule 2.1
    if device_type == "cpu":
        with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
            out = gla_chunk_attn(q, k, v)
    else:
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            out = gla_chunk_attn(q.cuda(), k.cuda(), v.cuda())

    assert torch.isfinite(out).all()


def test_telemetry_dispatch_backend() -> None:
    """Verify dispatcher fallback telemetry logs the backend used."""
    q = torch.randn(1, 2, 4, 16)
    k = torch.randn(1, 2, 4, 16)
    v = torch.randn(1, 2, 4, 16)

    with force_kernel_backend("pytorch"):
        out = gla_chunk_attn(q, k, v)
        assert out.shape == q.shape
        backend = get_last_dispatch_backend("gla_chunk_attn")
        assert backend == "pytorch"


def test_compliance_rules() -> None:
    """Verify static compliance with AGENTS.md rules."""
    base = Path("neuralops/ops/cuda/gla_chunk_attn")
    for p in base.glob("*"):
        if p.is_file() and p.suffix in {".py", ".cu", ".cpp", ".cuh", ".h"}:
            text = p.read_text(encoding="utf-8")
            assert "torch.cuda.amp" not in text, f"Deprecated torch.cuda.amp found in {p}"
            assert "time.time()" not in text, f"Non-CUDA timer time.time() found in {p}"
            assert "cuda:0" not in text, f"Hardcoded cuda:0 found in {p}"
            lines = len(text.splitlines())
            assert lines <= 800, f"File {p} exceeds 800 lines limit ({lines} lines)"
