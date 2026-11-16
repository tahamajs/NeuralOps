# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit and regression tests for HGRN2 Selective Scan CUDA operator and fallbacks (Wave 11 / W11-K16).

Validates:
1. Symbol exports, callability, and top-level namespace resolution.
2. CPU and reference fallback execution across shapes, dtypes, and dimensions.
3. Numerical sanity (no NaNs, no Infs, exact parity between operator and reference).
4. Analytical autograd backward pass and gradient flow through x and g.
5. Mixed requires_grad combinations (x only, g only, both).
6. AMP compatibility across float32 and bfloat16 using torch.amp.autocast.
7. Architectural compliance with AGENTS.md rules (no torch.cuda.amp, no time.time, file length limits).
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch
from torch.amp import autocast

import neuralops.ops.cuda as noc
from neuralops.ops.cuda.hgrn2_scan import (
    hgrn2_fused_bwd_scan,
    hgrn2_fused_bwd_scan_pytorch_ref,
    hgrn2_scan,
    hgrn2_scan_pytorch_ref,
)


def test_hgrn2_scan_symbol_exports() -> None:
    """Verify that all required symbols are exported and callable from package and top-level."""
    # Submodule exports
    assert callable(hgrn2_scan)
    assert callable(hgrn2_fused_bwd_scan)
    assert callable(hgrn2_fused_bwd_scan_pytorch_ref)
    assert callable(hgrn2_scan_pytorch_ref)

    # Top-level namespace exports
    assert callable(noc.hgrn2_scan)
    assert callable(noc.hgrn2_fused_bwd_scan)
    assert callable(noc.hgrn2_fused_bwd_scan_pytorch_ref)
    assert callable(noc.hgrn2_scan_pytorch_ref)

    # Registry / catalog listing
    assert "hgrn2_scan" in noc.list_cuda_kernels("modules")
    assert "hgrn2_scan" in noc.list_cuda_kernels("operators")
    assert "hgrn2_scan" in noc.list_cuda_kernels("all")


def test_hgrn2_scan_cpu_forward_basic() -> None:
    """Verify forward pass on CPU preserves tensor shapes, dtypes, and produces finite values."""
    torch.manual_seed(42)
    B, H, L, D = 2, 4, 8, 16
    x = torch.randn(B, H, L, D, dtype=torch.float32)
    g = torch.sigmoid(torch.randn(B, H, L, D, dtype=torch.float32))

    out = hgrn2_scan(x, g)

    assert isinstance(out, torch.Tensor)
    assert out.shape == (B, H, L, D)
    assert out.dtype == torch.float32
    assert out.device == x.device
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_hgrn2_scan_numerical_parity() -> None:
    """Verify that hgrn2_scan matches hgrn2_fused_bwd_scan_pytorch_ref exactly on CPU."""
    torch.manual_seed(1337)
    B, H, L, D = 2, 3, 10, 24
    x = torch.randn(B, H, L, D, dtype=torch.float32)
    g = torch.sigmoid(torch.randn(B, H, L, D, dtype=torch.float32))

    out_op = hgrn2_scan(x, g)
    out_ref = hgrn2_fused_bwd_scan_pytorch_ref(x, g)
    out_alias = hgrn2_scan_pytorch_ref(x, g)

    torch.testing.assert_close(out_op, out_ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(out_alias, out_ref, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("d_dim", [7, 16, 31, 64])
def test_hgrn2_scan_various_dimensions(d_dim: int) -> None:
    """Verify execution with D dimensions that are both divisible and non-divisible by 4."""
    torch.manual_seed(0)
    B, H, L = 1, 2, 6
    x = torch.randn(B, H, L, d_dim, dtype=torch.float32)
    g = torch.sigmoid(torch.randn(B, H, L, d_dim, dtype=torch.float32))

    out = noc.hgrn2_scan(x, g)
    ref = noc.hgrn2_fused_bwd_scan_pytorch_ref(x, g)

    assert out.shape == (B, H, L, d_dim)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_hgrn2_scan_autograd_both_gradients() -> None:
    """Verify backward gradient flow when both x and g require gradients."""
    torch.manual_seed(42)
    B, H, L, D = 2, 2, 6, 8
    x = torch.randn(B, H, L, D, dtype=torch.float32, requires_grad=True)
    g = torch.sigmoid(torch.randn(B, H, L, D, dtype=torch.float32)).detach().requires_grad_(True)

    out = hgrn2_scan(x, g)
    loss = (out**2).sum()
    loss.backward()

    assert x.grad is not None and x.grad.shape == x.shape
    assert g.grad is not None and g.grad.shape == g.shape
    assert not torch.isnan(x.grad).any()
    assert not torch.isnan(g.grad).any()
    assert not torch.isinf(x.grad).any()
    assert not torch.isinf(g.grad).any()
    assert (x.grad != 0).any().item()
    assert (g.grad != 0).any().item()


def test_hgrn2_scan_autograd_partial_gradients() -> None:
    """Verify autograd backward when only one input tensor requires gradient."""
    # Test only x requires grad
    torch.manual_seed(101)
    x = torch.randn(1, 2, 4, 8, dtype=torch.float32, requires_grad=True)
    g = torch.sigmoid(torch.randn(1, 2, 4, 8, dtype=torch.float32, requires_grad=False))

    out = hgrn2_scan(x, g)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert g.grad is None
    assert not torch.isnan(x.grad).any()

    # Test only g requires grad
    x2 = torch.randn(1, 2, 4, 8, dtype=torch.float32, requires_grad=False)
    g2 = torch.sigmoid(torch.randn(1, 2, 4, 8, dtype=torch.float32)).detach().requires_grad_(True)

    out2 = hgrn2_scan(x2, g2)
    loss2 = out2.sum()
    loss2.backward()

    assert x2.grad is None
    assert g2.grad is not None
    assert not torch.isnan(g2.grad).any()


def test_hgrn2_scan_autograd_numerical_parity() -> None:
    """Verify analytical autograd gradients match pure PyTorch autograd gradients."""
    torch.manual_seed(2024)
    B, H, L, D = 2, 2, 5, 8

    # Operator backward
    x1 = torch.randn(B, H, L, D, dtype=torch.float32, requires_grad=True)
    g1 = torch.sigmoid(torch.randn(B, H, L, D, dtype=torch.float32)).detach().requires_grad_(True)
    out1 = hgrn2_scan(x1, g1)
    loss1 = (out1**2).sum()
    loss1.backward()

    # Reference backward
    x2 = x1.detach().clone().requires_grad_(True)
    g2 = g1.detach().clone().requires_grad_(True)
    out2 = hgrn2_fused_bwd_scan_pytorch_ref(x2, g2)
    loss2 = (out2**2).sum()
    loss2.backward()

    assert x1.grad is not None and x2.grad is not None
    assert g1.grad is not None and g2.grad is not None
    torch.testing.assert_close(x1.grad, x2.grad, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(g1.grad, g2.grad, rtol=1e-5, atol=1e-5)


def test_hgrn2_scan_amp_autocast_compliance() -> None:
    """Verify torch.amp.autocast compatibility on CPU without torch.cuda.amp."""
    B, H, L, D = 1, 2, 4, 8
    x = torch.randn(B, H, L, D, dtype=torch.bfloat16)
    g = torch.sigmoid(torch.randn(B, H, L, D, dtype=torch.bfloat16))

    with autocast("cpu", dtype=torch.bfloat16):
        out = hgrn2_scan(x, g)

    assert out.shape == (B, H, L, D)
    assert not torch.isnan(out).any()


def test_hgrn2_scan_agents_compliance() -> None:
    """Verify compliance with AGENTS.md rules for code quality and anti-patterns."""
    base_dir = Path(__file__).resolve().parents[3] / "neuralops" / "ops" / "cuda" / "hgrn2_scan"
    assert base_dir.exists() and base_dir.is_dir()

    for p in base_dir.iterdir():
        if p.is_file():
            text = p.read_text(encoding="utf-8")
            lines = text.splitlines()

            # Rule 6.3: File size <= 800 lines
            assert len(lines) <= 800, f"{p.name} exceeds 800 lines ({len(lines)})"

            # Rule 2.1: No deprecated torch.cuda.amp
            assert "torch.cuda.amp" not in text, f"{p.name} contains deprecated torch.cuda.amp"

            # Rule 2.2: No time.time() for GPU timing
            assert "time.time()" not in text, f"{p.name} uses time.time()"

            # Rule 1.4 / 1.5: No main() entrypoints in library files
            assert "def main(" not in text, f"{p.name} contains def main()"
            assert '__name__ == "__main__"' not in text, f"{p.name} contains __main__ guard"

            # Rule 2.5: No hardcoded cuda:0
            assert "cuda:0" not in text, f"{p.name} contains hardcoded cuda:0"
