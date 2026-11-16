# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from pathlib import Path
import pytest
import torch

import neuralops.ops.cuda as noc
from neuralops.ops.cuda._fallback import get_last_dispatch_backend
from neuralops.ops.cuda.fwsvd import (
    FWSVDFunction,
    fwsvd_compress,
    fwsvd_compress_pytorch_ref,
    fwsvd_pytorch_ref,
)


def test_fwsvd_cpu_fallback():
    """Verify clean CPU fallback and numerical sanity for unweighted FWSVD."""
    W = torch.randn(64, 64)
    out = fwsvd_compress(W, rank=16)

    assert out.shape == W.shape
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_fwsvd_with_fisher_diag():
    """Verify FWSVD compression with 1D Fisher diagonal weighting."""
    M, N = 48, 32
    W = torch.randn(M, N)
    fisher_diag = torch.rand(N) + 0.1

    out = fwsvd_compress(W, fisher_diag=fisher_diag, rank=12)

    assert out.shape == W.shape
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_fwsvd_with_fisher_matrix():
    """Verify FWSVD compression with 2D Fisher information matrix."""
    M, N = 32, 32
    W = torch.randn(M, N)
    A = torch.randn(N, N)
    fisher_matrix = A.t() @ A + 0.05 * torch.eye(N)

    out = fwsvd_compress(W, fisher_matrix=fisher_matrix, rank=8)

    assert out.shape == W.shape
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_fwsvd_top_level_dispatch():
    """Verify symbols exported through neuralops.ops.cuda package lazy dispatcher."""
    W = torch.randn(32, 32)

    out_direct = noc.fwsvd(W, rank=8)
    out_compress = noc.fwsvd_compress(W, rank=8)
    out_ref = noc.fwsvd_pytorch_ref(W, rank=8)
    out_comp_ref = noc.fwsvd_compress_pytorch_ref(W, rank=8)

    assert out_direct.shape == W.shape
    assert out_compress.shape == W.shape
    assert out_ref.shape == W.shape
    assert out_comp_ref.shape == W.shape

    assert torch.allclose(out_direct, out_ref, atol=1e-5)
    assert torch.allclose(out_compress, out_comp_ref, atol=1e-5)


def test_fwsvd_autograd():
    """Verify analytical gradient computation through FWSVDFunction."""
    M, K, N = 16, 8, 24
    U = torch.randn(M, K, requires_grad=True)
    V = torch.randn(K, N, requires_grad=True)

    out = FWSVDFunction.apply(U, V)
    loss = (out**2).sum()
    loss.backward()

    assert U.grad is not None
    assert V.grad is not None
    assert not torch.isnan(U.grad).any()
    assert not torch.isnan(V.grad).any()
    assert not torch.isinf(U.grad).any()
    assert not torch.isinf(V.grad).any()

    # Verify analytical correctness against pure PyTorch matmul backward
    U_ref = U.detach().clone().requires_grad_(True)
    V_ref = V.detach().clone().requires_grad_(True)
    loss_ref = ((U_ref @ V_ref) ** 2).sum()
    loss_ref.backward()

    assert torch.allclose(U.grad, U_ref.grad, atol=1e-5)
    assert torch.allclose(V.grad, V_ref.grad, atol=1e-5)


def test_fwsvd_varying_shapes_and_ranks():
    """Verify execution across diverse matrix shapes, ranks, and data types."""
    test_cases = [
        ((16, 16), 4, torch.float32),
        ((32, 64), 16, torch.float32),
        ((64, 32), 8, torch.float64),
        ((8, 8), 1, torch.float32),
        ((20, 20), 40, torch.float32),  # rank > min(M, N)
    ]

    for (M, N), rank, dtype in test_cases:
        W = torch.randn(M, N, dtype=dtype)
        out = fwsvd_compress(W, rank=rank)
        assert out.shape == (M, N)
        assert out.dtype == dtype
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


def test_fwsvd_invalid_inputs():
    """Verify that invalid tensor ranks raise ValueError."""
    with pytest.raises(ValueError, match="requires a 2D weight matrix"):
        fwsvd_compress(torch.randn(32))

    with pytest.raises(ValueError, match="requires a 2D weight matrix"):
        fwsvd_compress(torch.randn(2, 16, 16))

    with pytest.raises(ValueError, match="requires a 2D weight matrix"):
        fwsvd_compress_pytorch_ref(torch.randn(10))


def test_fwsvd_dispatch_telemetry():
    """Verify telemetry tracking reports pytorch fallback on CPU."""
    W = torch.randn(16, 16)
    _ = fwsvd_compress(W, rank=4)
    backend = get_last_dispatch_backend("fwsvd")
    assert backend == "pytorch"


def test_fwsvd_compliance():
    """Verify AGENTS.md rules across all kernel source and python files."""
    base = Path("neuralops/ops/cuda/fwsvd")
    source_files = list(base.glob("*.*"))
    assert len(source_files) > 0, "No files found in kernel module directory"

    for p in source_files:
        if p.suffix in (".py", ".cu", ".cpp"):
            text = p.read_text(encoding="utf-8", errors="ignore")
            assert "torch.cuda.amp" not in text, f"Deprecated torch.cuda.amp in {p.name}"
            assert "time.time()" not in text, f"Non-compliant time.time() in {p.name}"
            assert "cuda:0" not in text, f"Hardcoded cuda:0 in {p.name}"
            assert len(text.splitlines()) <= 800, f"File {p.name} exceeds 800 lines"
