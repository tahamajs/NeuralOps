# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from pathlib import Path
import pytest
import torch

import neuralops.ops.cuda as noc
from neuralops.ops.cuda.gla_unnormalized_scan import (
    gla_unnormalized_scan,
    gla_unnormalized_scan_pytorch_ref,
)


def test_gla_unnormalized_scan_symbol_exports():
    """Verify clean symbol exports and callability from package and top-level."""
    assert hasattr(noc, "gla_unnormalized_scan")
    assert hasattr(noc, "gla_unnormalized_scan_pytorch_ref")
    assert callable(noc.gla_unnormalized_scan)
    assert callable(noc.gla_unnormalized_scan_pytorch_ref)
    assert callable(gla_unnormalized_scan)
    assert callable(gla_unnormalized_scan_pytorch_ref)


def test_gla_unnormalized_scan_cpu_fallback_4d_decay():
    """Verify CPU execution and numerical sanity with 4D decay."""
    B, H, N, D = 2, 4, 16, 32
    torch.manual_seed(42)
    q = torch.randn(B, H, N, D)
    k = torch.randn(B, H, N, D)
    v = torch.randn(B, H, N, D)
    decay = torch.randn(B, H, N, D) * 0.1

    out = gla_unnormalized_scan(q, k, v, decay)
    ref = gla_unnormalized_scan_pytorch_ref(q, k, v, decay)

    assert out.shape == (B, H, N, D)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_gla_unnormalized_scan_cpu_fallback_3d_decay():
    """Verify CPU execution with 3D decay (B, H, N)."""
    B, H, N, D = 2, 4, 8, 16
    torch.manual_seed(42)
    q = torch.randn(B, H, N, D)
    k = torch.randn(B, H, N, D)
    v = torch.randn(B, H, N, D)
    decay = torch.randn(B, H, N) * 0.1

    out = gla_unnormalized_scan(q, k, v, decay)
    ref = gla_unnormalized_scan_pytorch_ref(q, k, v, decay)

    assert out.shape == (B, H, N, D)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_gla_unnormalized_scan_cpu_fallback_varied_decay_shapes():
    """Verify CPU execution with 1D and 2D decay shapes."""
    B, H, N, D = 2, 4, 8, 16
    torch.manual_seed(42)
    q = torch.randn(B, H, N, D)
    k = torch.randn(B, H, N, D)
    v = torch.randn(B, H, N, D)

    # 1D decay (H,)
    decay_1d = torch.randn(H) * 0.1
    out_1d = gla_unnormalized_scan(q, k, v, decay_1d)
    assert out_1d.shape == (B, H, N, D)
    assert not torch.isnan(out_1d).any()

    # 2D decay (B, H)
    decay_2d = torch.randn(B, H) * 0.1
    out_2d = gla_unnormalized_scan(q, k, v, decay_2d)
    assert out_2d.shape == (B, H, N, D)
    assert not torch.isnan(out_2d).any()


def test_gla_unnormalized_scan_top_level_dispatch():
    """Verify top-level module noc.gla_unnormalized_scan dispatch."""
    B, H, N, D = 1, 2, 8, 16
    torch.manual_seed(42)
    q = torch.randn(B, H, N, D)
    k = torch.randn(B, H, N, D)
    v = torch.randn(B, H, N, D)
    decay = torch.randn(B, H, N, D) * 0.1

    out = noc.gla_unnormalized_scan(q, k, v, decay)
    assert out.shape == (B, H, N, D)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_gla_unnormalized_scan_autograd():
    """Verify analytical autograd backward produces valid gradients."""
    B, H, N, D = 2, 2, 4, 8
    torch.manual_seed(42)
    q = torch.randn(B, H, N, D, requires_grad=True)
    k = torch.randn(B, H, N, D, requires_grad=True)
    v = torch.randn(B, H, N, D, requires_grad=True)
    decay = torch.randn(B, H, N, D, requires_grad=True)

    out = gla_unnormalized_scan(q, k, v, decay)
    loss = out.sum()
    loss.backward()

    for tensor, name in [(q, "q"), (k, "k"), (v, "v"), (decay, "decay")]:
        assert tensor.grad is not None, f"Gradient missing for {name}"
        assert not torch.isnan(tensor.grad).any(), f"NaN in gradient for {name}"
        assert not torch.isinf(tensor.grad).any(), f"Inf in gradient for {name}"


def test_gla_unnormalized_scan_dtypes():
    """Verify execution across float32, bfloat16, float16."""
    B, H, N, D = 1, 2, 4, 8
    torch.manual_seed(42)

    for dtype in (torch.float32, torch.bfloat16, torch.float16):
        q = torch.randn(B, H, N, D, dtype=dtype)
        k = torch.randn(B, H, N, D, dtype=dtype)
        v = torch.randn(B, H, N, D, dtype=dtype)
        decay = (torch.randn(B, H, N, D) * 0.1).to(dtype)

        out = gla_unnormalized_scan(q, k, v, decay)
        assert out.shape == (B, H, N, D)
        assert out.dtype == dtype
        assert not torch.isnan(out).any()


def test_gla_unnormalized_scan_compliance():
    """Audit compliance with AGENTS.md rules."""
    base = Path("neuralops/ops/cuda/gla_unnormalized_scan")
    for p in base.glob("*"):
        if p.is_file():
            text = p.read_text(encoding="utf-8")
            assert "torch.cuda.amp" not in text, f"Rule 2.1 violation in {p}"
            assert "time.time()" not in text, f"Rule 2.2 violation in {p}"
            assert "cuda:0" not in text, f"Hardcoded cuda:0 violation in {p}"
            assert len(text.splitlines()) <= 800, f"Rule 6.3 line count violation in {p}"
