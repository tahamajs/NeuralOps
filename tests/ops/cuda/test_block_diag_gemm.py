# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for Block-Diagonal GEMM operator & CPU/PyTorch fallback (Wave 9 / W9-K12).
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import pytest
import torch
from torch.amp import autocast

import neuralops.ops.cuda as noc
from neuralops.ops.cuda.block_diag_gemm import block_diag_matmul, block_diag_matmul_pytorch_ref


def test_block_diag_gemm_cpu_basic() -> None:
    """Verify block_diag_matmul executes on CPU and produces expected shape and dtype."""
    # M=4*16=64, G=4, Kg=32 -> x feature dim = 128
    # G=4, Ng=16, Kg=32 -> output feature dim = G*Ng = 64
    x = torch.randn(4, 16, 128, dtype=torch.float32)
    weights = torch.randn(4, 16, 32, dtype=torch.float32)

    out = block_diag_matmul(x, weights)

    assert isinstance(out, torch.Tensor)
    assert out.shape == (4, 16, 64)
    assert out.dtype == torch.float32
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_block_diag_gemm_cpu_with_bias() -> None:
    """Verify block_diag_matmul operates correctly with bias on CPU."""
    x = torch.randn(2, 8, 64, dtype=torch.float32)
    weights = torch.randn(2, 16, 32, dtype=torch.float32)
    bias = torch.randn(2, 16, dtype=torch.float32)

    out = block_diag_matmul(x, weights, bias=bias)

    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 8, 32)
    assert not torch.isnan(out).any()


def test_block_diag_gemm_numerical_parity() -> None:
    """Verify output of dispatched block_diag_matmul matches pure PyTorch reference."""
    x = torch.randn(3, 7, 96, dtype=torch.float32)
    # G=3, Kg=32, Ng=24
    weights = torch.randn(3, 24, 32, dtype=torch.float32)
    bias = torch.randn(3, 24, dtype=torch.float32)

    ref_out = block_diag_matmul_pytorch_ref(x, weights, bias=bias)
    dispatched_out = block_diag_matmul(x, weights, bias=bias)

    torch.testing.assert_close(dispatched_out, ref_out, rtol=1e-5, atol=1e-5)


def test_block_diag_gemm_autograd_gradients() -> None:
    """Verify analytical autograd gradients flow properly back to inputs, weights, and bias."""
    x = torch.randn(2, 4, 32, dtype=torch.float32, requires_grad=True)
    weights = torch.randn(2, 8, 16, dtype=torch.float32, requires_grad=True)
    bias = torch.randn(2, 8, dtype=torch.float32, requires_grad=True)

    out = block_diag_matmul(x, weights, bias=bias)
    loss = (out**2).sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert not torch.isnan(x.grad).any()

    assert weights.grad is not None
    assert weights.grad.shape == weights.shape
    assert not torch.isnan(weights.grad).any()

    assert bias.grad is not None
    assert bias.grad.shape == bias.shape
    assert not torch.isnan(bias.grad).any()


def test_block_diag_gemm_amp_compliance() -> None:
    """Verify torch.amp compliance (Rule 2.1) across float16 and bfloat16."""
    # Native float16
    x_fp16 = torch.randn(2, 4, 32, dtype=torch.float16)
    w_fp16 = torch.randn(2, 8, 16, dtype=torch.float16)
    out_fp16 = block_diag_matmul(x_fp16, w_fp16)
    assert out_fp16.dtype == torch.float16
    assert not torch.isnan(out_fp16).any()

    # Native bfloat16
    x_bf16 = torch.randn(2, 4, 32, dtype=torch.bfloat16)
    w_bf16 = torch.randn(2, 8, 16, dtype=torch.bfloat16)
    out_bf16 = block_diag_matmul(x_bf16, w_bf16)
    assert out_bf16.dtype == torch.bfloat16
    assert not torch.isnan(out_bf16).any()

    # torch.amp.autocast context
    with autocast("cpu", dtype=torch.bfloat16):
        x = torch.randn(2, 4, 32)
        w = torch.randn(2, 8, 16)
        out = block_diag_matmul(x, w)
        assert isinstance(out, torch.Tensor)
        assert not torch.isnan(out).any()


def test_block_diag_gemm_top_level_symbol_resolution() -> None:
    """Verify top-level symbol resolution via neuralops.ops.cuda."""
    x = torch.randn(2, 4, 32, dtype=torch.float32)
    weights = torch.randn(2, 8, 16, dtype=torch.float32)

    # Directly through module export
    out1 = noc.block_diag_gemm(x, weights)
    assert out1.shape == (2, 4, 16)

    # Through mapped name block_diag_matmul
    out2 = noc.block_diag_matmul(x, weights)
    assert out2.shape == (2, 4, 16)
    torch.testing.assert_close(out1, out2)


def test_block_diag_gemm_higher_dimensions() -> None:
    """Verify block_diag_matmul supports arbitrary batch prefix dimensions (e.g., 4D tensor)."""
    # Batch dimensions (2, 3, 4), G=2, Kg=8 -> total feature dim = 16
    x = torch.randn(2, 3, 4, 16, dtype=torch.float32)
    weights = torch.randn(2, 12, 8, dtype=torch.float32)

    out = block_diag_matmul(x, weights)
    assert out.shape == (2, 3, 4, 24)
    assert not torch.isnan(out).any()


def test_no_forbidden_patterns_in_block_diag_gemm() -> None:
    """Verify code hygiene: absence of deprecated torch.cuda.amp, raw time.time(), etc."""
    pkg_dir = Path("neuralops/ops/cuda/block_diag_gemm")
    for fpath in pkg_dir.glob("*.py"):
        src = fpath.read_text(encoding="utf-8")
        assert "torch.cuda.amp" not in src, f"Deprecated torch.cuda.amp found in {fpath}"
        assert "time.time()" not in src, f"Unsynchronized time.time() found in {fpath}"


def test_file_line_limits_rule_6_3() -> None:
    """Verify all source files in block_diag_gemm are within Rule 6.3 limits (<=800 lines)."""
    pkg_dir = Path("neuralops/ops/cuda/block_diag_gemm")
    for fpath in pkg_dir.glob("*"):
        if fpath.is_file() and fpath.suffix in [".py", ".cpp", ".cu"]:
            line_count = len(fpath.read_text(encoding="utf-8").splitlines())
            assert line_count <= 800, f"File {fpath} has {line_count} lines (> 800)"
