# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit and regression tests for HGRN2 Spatial Scan CUDA kernel and CPU fallback.
Strictly follows AGENTS.md rules.
"""

from __future__ import annotations

from pathlib import Path
import pytest
import torch

import neuralops.ops.cuda as noc
from neuralops.ops.cuda.hgrn2_spatial_scan import (
    hgrn2_spatial_scan,
    hgrn2_spatial_scan_pytorch_ref,
)


def test_hgrn2_spatial_scan_symbol_exports() -> None:
    """Verify clean symbol exports and top-level integration in neuralops.ops.cuda."""
    assert hasattr(noc, "hgrn2_spatial_scan")
    assert hasattr(noc, "hgrn2_spatial_scan_pytorch_ref")
    assert callable(noc.hgrn2_spatial_scan)
    assert callable(noc.hgrn2_spatial_scan_pytorch_ref)
    assert callable(hgrn2_spatial_scan)
    assert callable(hgrn2_spatial_scan_pytorch_ref)
    assert "hgrn2_spatial_scan" in noc.__all__
    assert "hgrn2_spatial_scan_pytorch_ref" in noc.__all__


def test_hgrn2_spatial_scan_cpu_fallback_2args() -> None:
    """Verify 2-argument CPU fallback execution and numerical sanity (no NaNs, no Infs)."""
    B, H, N, D = 2, 4, 8, 16
    x = torch.randn(B, H, N, D, dtype=torch.float32)
    gate = torch.randn(B, H, N, D, dtype=torch.float32)

    out = hgrn2_spatial_scan(x, gate)

    assert isinstance(out, torch.Tensor)
    assert out.shape == (B, H, N, D)
    assert out.dtype == torch.float32
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()

    # Exact parity against reference implementation
    ref = hgrn2_spatial_scan_pytorch_ref(x, gate)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_hgrn2_spatial_scan_cpu_fallback_3d_gate() -> None:
    """Verify CPU fallback with 3D gate tensor (B, H, N) expanded across feature dim D."""
    B, H, N, D = 2, 4, 8, 16
    x = torch.randn(B, H, N, D, dtype=torch.float32)
    gate = torch.randn(B, H, N, dtype=torch.float32)

    out = hgrn2_spatial_scan(x, gate)

    assert out.shape == (B, H, N, D)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()

    ref = hgrn2_spatial_scan_pytorch_ref(x, gate)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_hgrn2_spatial_scan_cpu_fallback_4args() -> None:
    """Verify 4-argument CPU fallback execution (q, k, v, gate) and numerical sanity."""
    B, H, N, D = 2, 3, 6, 12
    q = torch.randn(B, H, N, D, dtype=torch.float32)
    k = torch.randn(B, H, N, D, dtype=torch.float32)
    v = torch.randn(B, H, N, D, dtype=torch.float32)
    gate = torch.randn(B, H, N, D, dtype=torch.float32)

    out = hgrn2_spatial_scan(q, k, v, gate)

    assert out.shape == (B, H, N, D)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()

    ref = hgrn2_spatial_scan_pytorch_ref(q, k, v, gate)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_hgrn2_spatial_scan_top_level_dispatch() -> None:
    """Verify top-level noc.hgrn2_spatial_scan call matches direct import."""
    B, H, N, D = 1, 2, 4, 8
    x = torch.randn(B, H, N, D)
    gate = torch.randn(B, H, N, D)

    out_direct = hgrn2_spatial_scan(x, gate)
    out_noc = noc.hgrn2_spatial_scan(x, gate)

    torch.testing.assert_close(out_direct, out_noc)


def test_hgrn2_spatial_scan_keyword_args() -> None:
    """Verify execution with keyword arguments."""
    B, H, N, D = 1, 2, 4, 8
    x = torch.randn(B, H, N, D)
    g = torch.randn(B, H, N, D)

    out = hgrn2_spatial_scan(x=x, g=g)
    ref = hgrn2_spatial_scan_pytorch_ref(x=x, gate=g)
    torch.testing.assert_close(out, ref)


def test_hgrn2_spatial_scan_autograd_backward() -> None:
    """Verify autograd backward gradient propagation with both full and partial requires_grad."""
    B, H, N, D = 2, 2, 4, 8

    # Partial requires_grad (only x requires grad)
    x = torch.randn(B, H, N, D, requires_grad=True)
    g = torch.randn(B, H, N, D, requires_grad=False)
    out = hgrn2_spatial_scan(x, g)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert g.grad is None
    assert not torch.isnan(x.grad).any()
    assert not torch.isinf(x.grad).any()

    # Full requires_grad (both x and g require grad)
    x2 = torch.randn(B, H, N, D, requires_grad=True)
    g2 = torch.randn(B, H, N, D, requires_grad=True)
    out2 = hgrn2_spatial_scan(x2, g2)
    loss2 = (out2 * 2.0).sum()
    loss2.backward()

    assert x2.grad is not None
    assert g2.grad is not None
    assert not torch.isnan(x2.grad).any()
    assert not torch.isnan(g2.grad).any()

    # 4-argument autograd backward
    q = torch.randn(B, H, N, D, requires_grad=True)
    k = torch.randn(B, H, N, D, requires_grad=True)
    v = torch.randn(B, H, N, D, requires_grad=True)
    g3 = torch.randn(B, H, N, D, requires_grad=True)
    out4 = hgrn2_spatial_scan(q, k, v, g3)
    loss4 = out4.sum()
    loss4.backward()

    for t, name in [(q, "q"), (k, "k"), (v, "v"), (g3, "g")]:
        assert t.grad is not None, f"Gradient for {name} was not computed"
        assert not torch.isnan(t.grad).any(), f"Gradient for {name} contains NaN"


def test_hgrn2_spatial_scan_invalid_args() -> None:
    """Verify appropriate ValueError on invalid argument counts."""
    x = torch.randn(2, 2, 4, 8)
    with pytest.raises(ValueError, match="expected 2 or 4 arguments"):
        hgrn2_spatial_scan(x)

    with pytest.raises(ValueError, match="expected 2 or 4 arguments"):
        hgrn2_spatial_scan(x, x, x)


def test_hgrn2_spatial_scan_compliance() -> None:
    """Verify AGENTS.md rules compliance for hgrn2_spatial_scan files."""
    base = Path("neuralops/ops/cuda/hgrn2_spatial_scan")
    assert base.exists(), f"Directory {base} does not exist"

    for file_path in base.iterdir():
        if file_path.suffix in (".py", ".cu", ".cpp"):
            text = file_path.read_text(encoding="utf-8")
            lines = text.splitlines()

            # Rule 6.3 - Line count <= 800
            assert len(lines) <= 800, f"{file_path.name} exceeds 800 lines ({len(lines)})"

            # Rule 2.1 - No deprecated torch.cuda.amp
            assert "torch.cuda.amp" not in text, f"Deprecated torch.cuda.amp in {file_path.name}"

            # Rule 2.2 - No time.time() for GPU timing
            assert "time.time()" not in text, f"Unsynchronized time.time() in {file_path.name}"

            # Hardcoded device check
            assert "cuda:0" not in text, f"Hardcoded cuda:0 found in {file_path.name}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_hgrn2_spatial_scan_cuda_execution() -> None:
    """Verify CUDA execution when GPU is available."""
    device = torch.device("cuda")
    B, H, N, D = 2, 4, 8, 16
    x = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
    gate = torch.randn(B, H, N, D, device=device, dtype=torch.float32)

    out = hgrn2_spatial_scan(x, gate)
    assert out.is_cuda
    assert out.shape == (B, H, N, D)
    assert not torch.isnan(out).any()
