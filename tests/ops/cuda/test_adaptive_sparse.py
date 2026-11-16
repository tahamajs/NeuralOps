# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for CUDA Adaptive Sparse Attention Operator and Fallbacks (Wave 9 W9-K3).

Verifies:
1. Operator importability and symbol resolution from neuralops.ops.cuda.adaptive_sparse and neuralops.ops.cuda.
2. Fallback execution on CPU / non-CUDA environments.
3. Tensor shape contract preservation and numerical stability (no NaNs, no Infs).
4. Gradient flow and backward pass autograd differentiability.
5. PyTorch AMP (torch.amp.autocast) compliance across float16/bfloat16.
6. File length and code hygiene constraints (<= 800 lines, no deprecated torch.cuda.amp, no hardcoded cuda:0).
"""

from __future__ import annotations

import math
from pathlib import Path
import pytest
import torch
from torch.amp import autocast

import neuralops.ops.cuda as noc
from neuralops.ops.cuda.adaptive_sparse import (
    AdaptiveSparseAttentionFunction,
    adaptive_sparse_attention_forward,
)


def test_imports_and_symbols() -> None:
    """Verify that adaptive_sparse and adaptive_sparse_attention_forward are available."""
    assert hasattr(noc, "adaptive_sparse"), "noc missing adaptive_sparse"
    assert hasattr(noc, "adaptive_sparse_attention_forward"), "noc missing adaptive_sparse_attention_forward"
    assert callable(noc.adaptive_sparse)
    assert callable(adaptive_sparse_attention_forward)
    assert issubclass(AdaptiveSparseAttentionFunction, torch.autograd.Function)


def test_cpu_fallback_forward_shape_and_values() -> None:
    """Verify CPU fallback forward pass shape contract and finite numerical values."""
    B, H, L, D = 2, 4, 16, 32
    q = torch.randn(B, H, L, D, device="cpu")
    k = torch.randn(B, H, L, D, device="cpu")
    v = torch.randn(B, H, L, D, device="cpu")

    out = adaptive_sparse_attention_forward(q, k, v, sparsity=0.5)
    assert out.shape == (B, H, L, D), f"Expected shape {(B, H, L, D)}, got {out.shape}"
    assert not torch.isnan(out).any(), "NaN found in output"
    assert not torch.isinf(out).any(), "Inf found in output"


def test_adaptive_sparse_noc_entrypoint() -> None:
    """Verify neuralops.ops.cuda.adaptive_sparse entrypoint invocation."""
    B, H, L, D = 1, 2, 8, 16
    q = torch.randn(B, H, L, D, device="cpu")
    k = torch.randn(B, H, L, D, device="cpu")
    v = torch.randn(B, H, L, D, device="cpu")

    out = noc.adaptive_sparse(q, k, v)
    assert out.shape == q.shape
    assert not torch.isnan(out).any()


def test_backward_gradient_flow() -> None:
    """Verify autograd backward gradient flow through adaptive_sparse_attention_forward."""
    B, H, L, D = 2, 2, 8, 16
    q = torch.randn(B, H, L, D, device="cpu", requires_grad=True)
    k = torch.randn(B, H, L, D, device="cpu", requires_grad=True)
    v = torch.randn(B, H, L, D, device="cpu", requires_grad=True)

    out = adaptive_sparse_attention_forward(q, k, v, sparsity=0.5)
    loss = out.sum()
    loss.backward()

    assert q.grad is not None, "q.grad is None after backward"
    assert k.grad is not None, "k.grad is None after backward"
    assert v.grad is not None, "v.grad is None after backward"
    assert q.grad.shape == q.shape
    assert k.grad.shape == k.shape
    assert v.grad.shape == v.shape
    assert not torch.isnan(q.grad).any(), "NaN in q.grad"
    assert not torch.isnan(k.grad).any(), "NaN in k.grad"
    assert not torch.isnan(v.grad).any(), "NaN in v.grad"


def test_torch_amp_autocast_cpu() -> None:
    """Verify torch.amp.autocast compatibility with bfloat16 on CPU."""
    B, H, L, D = 2, 2, 8, 16
    q = torch.randn(B, H, L, D, device="cpu")
    k = torch.randn(B, H, L, D, device="cpu")
    v = torch.randn(B, H, L, D, device="cpu")

    with autocast(device_type="cpu", dtype=torch.bfloat16):
        out = adaptive_sparse_attention_forward(q, k, v, sparsity=0.6)

    assert out.dtype == torch.bfloat16, f"Expected bfloat16 output under autocast, got {out.dtype}"
    assert out.shape == (B, H, L, D)
    assert not torch.isnan(out).any()


@pytest.mark.parametrize("sparsity", [0.0, 0.25, 0.5, 0.75, 0.9])
def test_various_sparsity_levels(sparsity: float) -> None:
    """Verify execution across different sparsity levels."""
    B, H, L, D = 1, 2, 16, 32
    q = torch.randn(B, H, L, D, device="cpu")
    k = torch.randn(B, H, L, D, device="cpu")
    v = torch.randn(B, H, L, D, device="cpu")

    out = adaptive_sparse_attention_forward(q, k, v, sparsity=sparsity)
    assert out.shape == (B, H, L, D)
    assert not torch.isnan(out).any()


def test_file_size_and_hygiene_governance() -> None:
    """Verify governance invariants: <= 800 lines per file, no deprecated torch.cuda.amp, no cuda:0."""
    kernel_dir = Path("neuralops/ops/cuda/adaptive_sparse")
    py_files = list(kernel_dir.glob("*.py"))
    cpp_files = list(kernel_dir.glob("*.cpp"))
    cu_files = list(kernel_dir.glob("*.cu"))
    all_files = py_files + cpp_files + cu_files

    assert len(all_files) >= 3, "Expected at least 3 source files in adaptive_sparse"

    for file_path in all_files:
        content = file_path.read_text(encoding="utf-8")
        lines = content.splitlines()
        # Rule 6.3: <= 800 lines hard limit
        assert len(lines) <= 800, f"{file_path} exceeds 800 lines ({len(lines)} lines)"

        # Rule 2.1 & Rule 10: No torch.cuda.amp
        assert "torch.cuda.amp" not in content, f"{file_path} contains forbidden torch.cuda.amp"

        # Rule 10: No hardcoded cuda:0
        assert "cuda:0" not in content, f"{file_path} contains hardcoded cuda:0"
