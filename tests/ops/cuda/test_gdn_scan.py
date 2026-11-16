# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit and regression tests for Gated DeltaNet Chunked Scan CUDA operator and fallback (Wave 11 / W11-K9).
"""

from __future__ import annotations

import inspect
from pathlib import Path
import pytest
import torch
from torch.amp import autocast

import neuralops.ops.cuda as noc
from neuralops.ops.cuda.gdn_scan import (
    _GDNScanFunction,
    chunked_delta_gate_scan,
    chunked_delta_gate_scan_pytorch_ref,
    gdn_scan,
    gdn_scan_pytorch_ref,
)


def test_gdn_scan_symbol_exports_and_catalog() -> None:
    """Verify that all required symbols are exported and cataloged properly."""
    assert callable(chunked_delta_gate_scan)
    assert callable(chunked_delta_gate_scan_pytorch_ref)
    assert callable(gdn_scan)
    assert callable(gdn_scan_pytorch_ref)

    assert hasattr(noc, "chunked_delta_gate_scan")
    assert hasattr(noc, "chunked_delta_gate_scan_pytorch_ref")
    assert hasattr(noc, "gdn_scan")
    assert hasattr(noc, "gdn_scan_pytorch_ref")

    assert "gdn_scan" in noc.list_cuda_kernels("modules")
    assert "gdn_scan" in noc.CUDA_KERNEL_CATALOG


def test_gdn_scan_cpu_forward_shapes() -> None:
    """Verify forward operator executes on CPU and preserves correct output shapes."""
    B, H, L, D_K, D_V = 2, 4, 16, 32, 32
    q = torch.randn(B, H, L, D_K, dtype=torch.float32)
    k = torch.randn(B, H, L, D_K, dtype=torch.float32)
    v = torch.randn(B, H, L, D_V, dtype=torch.float32)
    alpha = torch.sigmoid(torch.randn(B, H, L, dtype=torch.float32))
    beta = torch.sigmoid(torch.randn(B, H, L, dtype=torch.float32))

    out = chunked_delta_gate_scan(q, k, v, alpha, beta)

    assert isinstance(out, torch.Tensor)
    assert out.shape == (B, H, L, D_V)
    assert out.dtype == torch.float32
    assert out.device == q.device
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_gdn_scan_cpu_asymmetric_dimensions() -> None:
    """Verify execution with asymmetric key and value dimensions (D_K != D_V)."""
    B, H, L, D_K, D_V = 1, 2, 8, 16, 32
    q = torch.randn(B, H, L, D_K, dtype=torch.float32)
    k = torch.randn(B, H, L, D_K, dtype=torch.float32)
    v = torch.randn(B, H, L, D_V, dtype=torch.float32)
    alpha = torch.sigmoid(torch.randn(B, H, L, dtype=torch.float32))
    beta = torch.sigmoid(torch.randn(B, H, L, dtype=torch.float32))

    out = gdn_scan(q, k, v, alpha, beta)
    assert out.shape == (B, H, L, D_V)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_gdn_scan_numerical_parity_vs_pytorch_ref() -> None:
    """Verify forward matches chunked_delta_gate_scan_pytorch_ref exactly on CPU."""
    B, H, L, D_K, D_V = 2, 2, 12, 16, 16
    q = torch.randn(B, H, L, D_K, dtype=torch.float32)
    k = torch.randn(B, H, L, D_K, dtype=torch.float32)
    v = torch.randn(B, H, L, D_V, dtype=torch.float32)
    alpha = torch.sigmoid(torch.randn(B, H, L, dtype=torch.float32))
    beta = torch.sigmoid(torch.randn(B, H, L, dtype=torch.float32))

    out_op = chunked_delta_gate_scan(q, k, v, alpha, beta)
    out_ref = chunked_delta_gate_scan_pytorch_ref(q, k, v, alpha, beta)

    torch.testing.assert_close(out_op, out_ref, rtol=1e-5, atol=1e-5)


def test_gdn_scan_autograd_function_direct_apply() -> None:
    """Verify _GDNScanFunction.apply executes directly on CPU."""
    B, H, L, D_K, D_V = 1, 1, 6, 8, 8
    q = torch.randn(B, H, L, D_K, dtype=torch.float32)
    k = torch.randn(B, H, L, D_K, dtype=torch.float32)
    v = torch.randn(B, H, L, D_V, dtype=torch.float32)
    alpha = torch.sigmoid(torch.randn(B, H, L, dtype=torch.float32))
    beta = torch.sigmoid(torch.randn(B, H, L, dtype=torch.float32))

    out = _GDNScanFunction.apply(q, k, v, alpha, beta)
    assert out.shape == (B, H, L, D_V)
    assert not torch.isnan(out).any()


def test_gdn_scan_autograd_backward() -> None:
    """Verify analytical gradients are computed for all inputs through autograd backward."""
    B, H, L, D_K, D_V = 2, 2, 6, 8, 8
    q = torch.randn(B, H, L, D_K, dtype=torch.float32, requires_grad=True)
    k = torch.randn(B, H, L, D_K, dtype=torch.float32, requires_grad=True)
    v = torch.randn(B, H, L, D_V, dtype=torch.float32, requires_grad=True)
    alpha = torch.randn(B, H, L, dtype=torch.float32, requires_grad=True)
    beta = torch.randn(B, H, L, dtype=torch.float32, requires_grad=True)

    out = chunked_delta_gate_scan(q, k, v, alpha, beta)
    loss = (out**2).sum()
    loss.backward()

    assert q.grad is not None and q.grad.shape == q.shape
    assert k.grad is not None and k.grad.shape == k.shape
    assert v.grad is not None and v.grad.shape == v.shape
    assert alpha.grad is not None and alpha.grad.shape == alpha.shape
    assert beta.grad is not None and beta.grad.shape == beta.shape

    assert (q.grad != 0).any().item()
    assert (k.grad != 0).any().item()
    assert (v.grad != 0).any().item()

    assert not torch.isnan(q.grad).any()
    assert not torch.isnan(k.grad).any()
    assert not torch.isnan(v.grad).any()
    assert not torch.isnan(alpha.grad).any()
    assert not torch.isnan(beta.grad).any()


def test_gdn_scan_amp_autocast_compliance() -> None:
    """Verify torch.amp.autocast compatibility without deprecated torch.cuda.amp."""
    B, H, L, D_K, D_V = 1, 2, 8, 16, 16
    q = torch.randn(B, H, L, D_K, dtype=torch.bfloat16)
    k = torch.randn(B, H, L, D_K, dtype=torch.bfloat16)
    v = torch.randn(B, H, L, D_V, dtype=torch.bfloat16)
    alpha = torch.sigmoid(torch.randn(B, H, L, dtype=torch.bfloat16))
    beta = torch.sigmoid(torch.randn(B, H, L, dtype=torch.bfloat16))

    with autocast("cpu", dtype=torch.bfloat16):
        out = chunked_delta_gate_scan(q, k, v, alpha, beta)

    assert out.shape == (B, H, L, D_V)
    assert out.dtype == torch.bfloat16
    assert not torch.isnan(out).any()


def test_gdn_scan_top_level_noc_dispatch() -> None:
    """Verify neuralops.ops.cuda aliases produce consistent outputs."""
    B, H, L, D_K, D_V = 1, 2, 8, 16, 16
    q = torch.randn(B, H, L, D_K)
    k = torch.randn(B, H, L, D_K)
    v = torch.randn(B, H, L, D_V)
    alpha = torch.sigmoid(torch.randn(B, H, L))
    beta = torch.sigmoid(torch.randn(B, H, L))

    out_alias = noc.gdn_scan(q, k, v, alpha, beta)
    out_canonical = noc.chunked_delta_gate_scan(q, k, v, alpha, beta)
    out_ref_alias = noc.gdn_scan_pytorch_ref(q, k, v, alpha, beta)
    out_ref_canonical = noc.chunked_delta_gate_scan_pytorch_ref(q, k, v, alpha, beta)

    torch.testing.assert_close(out_alias, out_canonical)
    torch.testing.assert_close(out_ref_alias, out_ref_canonical)
    torch.testing.assert_close(out_alias, out_ref_canonical)


def test_gdn_scan_agents_compliance() -> None:
    """Verify AGENTS.md rules: line count <= 800, no torch.cuda.amp, no time.time, no cuda:0."""
    base_dir = Path("neuralops/ops/cuda/gdn_scan")
    assert base_dir.exists(), f"Directory not found: {base_dir}"

    for p in base_dir.glob("*"):
        if p.is_file() and p.suffix in (".py", ".cpp", ".cu", ".cuh"):
            content = p.read_text(encoding="utf-8")
            assert "torch.cuda.amp" not in content, f"Forbidden torch.cuda.amp in {p.name}"
            assert "time.time()" not in content, f"Forbidden time.time() in {p.name}"
            assert "cuda:0" not in content, f"Forbidden hardcoded cuda:0 in {p.name}"
            lines = content.splitlines()
            assert len(lines) <= 800, f"File {p.name} exceeds 800 lines ({len(lines)})"
