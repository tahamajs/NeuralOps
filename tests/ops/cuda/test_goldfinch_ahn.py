#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Unit and regression tests for goldfinch_ahn CUDA/CPU operator."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.amp import autocast

import neuralops.ops.cuda as noc
from neuralops.ops.cuda.goldfinch_ahn import (
    goldfinch_ahn,
    goldfinch_ahn_ref,
)


def test_goldfinch_ahn_symbol_exports() -> None:
    """Verify goldfinch_ahn symbols are exported cleanly in module and package namespace."""
    assert hasattr(noc, "goldfinch_ahn")
    assert hasattr(noc, "goldfinch_ahn_ref")
    assert callable(noc.goldfinch_ahn)
    assert callable(noc.goldfinch_ahn_ref)
    assert "goldfinch_ahn" in dir(noc)
    assert "goldfinch_ahn_ref" in dir(noc)
    assert "goldfinch_ahn" in noc.__all__
    assert "goldfinch_ahn_ref" in noc.__all__


def test_goldfinch_ahn_cpu_fallback_execution() -> None:
    """Verify goldfinch_ahn executes on CPU via reference fallback with finite values."""
    b, h, n, d = 2, 4, 16, 32
    q = torch.randn(b, h, n, d)
    k = torch.randn(b, h, n, d)
    v = torch.randn(b, h, n, d)
    beta = torch.sigmoid(torch.randn(b, h, n, d))

    out = goldfinch_ahn(q, k, v, beta)
    assert out.shape == (b, h, n, d)
    assert torch.isfinite(out).all()
    assert not torch.isnan(out).any()


def test_goldfinch_ahn_top_level_dispatch() -> None:
    """Verify dispatch through top-level neuralops.ops.cuda namespace."""
    b, h, n, d = 1, 2, 8, 16
    q = torch.randn(b, h, n, d)
    k = torch.randn(b, h, n, d)
    v = torch.randn(b, h, n, d)
    beta = torch.sigmoid(torch.randn(b, h, n, d))

    out = noc.goldfinch_ahn(q, k, v, beta)  # type: ignore[operator]
    assert out.shape == (b, h, n, d)
    assert torch.isfinite(out).all()


def test_goldfinch_ahn_parity_with_reference() -> None:
    """Verify numerical parity between goldfinch_ahn and goldfinch_ahn_ref."""
    b, h, n, d = 2, 2, 10, 16
    q = torch.randn(b, h, n, d, dtype=torch.float64)
    k = torch.randn(b, h, n, d, dtype=torch.float64)
    v = torch.randn(b, h, n, d, dtype=torch.float64)
    beta = torch.sigmoid(torch.randn(b, h, n, d, dtype=torch.float64))

    out_kernel = goldfinch_ahn(q, k, v, beta)
    out_ref = goldfinch_ahn_ref(q, k, v, beta)

    assert torch.allclose(out_kernel, out_ref, atol=1e-7, rtol=1e-7)


def test_goldfinch_ahn_autograd_gradients() -> None:
    """Verify backward gradient propagation through goldfinch_ahn on CPU."""
    b, h, n, d = 2, 2, 8, 16
    q = torch.randn(b, h, n, d, requires_grad=True)
    k = torch.randn(b, h, n, d, requires_grad=True)
    v = torch.randn(b, h, n, d, requires_grad=True)
    beta = torch.sigmoid(torch.randn(b, h, n, d)).requires_grad_(True)

    out = goldfinch_ahn(q, k, v, beta)
    loss = out.sum()
    loss.backward()

    assert q.grad is not None and torch.isfinite(q.grad).all()
    assert k.grad is not None and torch.isfinite(k.grad).all()
    assert v.grad is not None and torch.isfinite(v.grad).all()
    assert beta.grad is not None and torch.isfinite(beta.grad).all()


def test_goldfinch_ahn_amp_autocast_cpu() -> None:
    """Verify compatibility with torch.amp.autocast for CPU execution."""
    b, h, n, d = 2, 2, 8, 16
    q = torch.randn(b, h, n, d)
    k = torch.randn(b, h, n, d)
    v = torch.randn(b, h, n, d)
    beta = torch.sigmoid(torch.randn(b, h, n, d))

    with autocast(device_type="cpu", dtype=torch.bfloat16):
        out = goldfinch_ahn(q, k, v, beta)

    assert out.shape == (b, h, n, d)
    assert torch.isfinite(out).all()


def test_goldfinch_ahn_edge_cases() -> None:
    """Verify handling of edge cases: empty dimensions, single-step sequences, invalid shapes."""
    # Empty sequence length
    q_empty = torch.randn(2, 2, 0, 16)
    k_empty = torch.randn(2, 2, 0, 16)
    v_empty = torch.randn(2, 2, 0, 16)
    b_empty = torch.randn(2, 2, 0, 16)
    out_empty = goldfinch_ahn(q_empty, k_empty, v_empty, b_empty)
    assert out_empty.shape == (2, 2, 0, 16)

    # Single-step sequence
    q_single = torch.randn(1, 1, 1, 8)
    k_single = torch.randn(1, 1, 1, 8)
    v_single = torch.randn(1, 1, 1, 8)
    b_single = torch.sigmoid(torch.randn(1, 1, 1, 8))
    out_single = goldfinch_ahn(q_single, k_single, v_single, b_single)
    assert out_single.shape == (1, 1, 1, 8)
    assert torch.isfinite(out_single).all()

    # Shape mismatch error
    with pytest.raises(ValueError):
        goldfinch_ahn(torch.randn(2, 2, 8, 16), torch.randn(2, 2, 8, 32), v_single, b_single)


def test_goldfinch_ahn_compliance() -> None:
    """Audit compliance with AGENTS.md rules for goldfinch_ahn."""
    base = Path("neuralops/ops/cuda/goldfinch_ahn")
    for file_path in base.glob("*"):
        if file_path.is_file():
            text = file_path.read_text(encoding="utf-8")
            # Rule 6.3: Line count <= 800
            assert len(text.splitlines()) <= 800, f"{file_path} exceeds 800 lines"
            # Rule 2.1: No deprecated torch.cuda.amp
            assert "torch.cuda.amp" not in text, f"{file_path} contains torch.cuda.amp"
            # Rule 2.2: No time.time() for GPU timing
            assert "time.time()" not in text, f"{file_path} contains time.time()"
            # No hardcoded cuda:0
            assert "cuda:0" not in text, f"{file_path} contains hardcoded cuda:0"
