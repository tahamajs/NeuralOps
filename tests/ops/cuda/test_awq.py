# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for AWQ (Activation-aware Weight Quantization) CUDA kernel & CPU fallback (Wave 9 / W9-K8).
"""

from __future__ import annotations

import pytest
import torch
from torch.amp import autocast

import neuralops.ops.cuda as noc
from neuralops.ops.cuda.awq import AWQFunction, awq_compress
from neuralops.ops.triton.quantize_awq import _awq_pytorch, _compute_awq_scales


def test_awq_cpu_fallback_basic() -> None:
    """Verify AWQ compression executes properly on CPU fallback returning a valid tensor."""
    w = torch.randn(8, 16, dtype=torch.float32)
    out = awq_compress(w)

    assert isinstance(out, torch.Tensor)
    assert not isinstance(out, tuple)
    assert out.shape == w.shape
    assert out.dtype == w.dtype
    assert out.device == w.device
    assert torch.isfinite(out).all().item()


def test_awq_cpu_numerical_parity() -> None:
    """Verify AWQ CUDA module CPU fallback matches the PyTorch reference implementation."""
    w = torch.randn(16, 32, dtype=torch.float32)
    act_scales = torch.rand(32).clamp(min=0.1)

    out = awq_compress(w, group_size=8, bits=4, act_scales=act_scales, alpha=0.5)
    ref = _awq_pytorch(w, group_size=8, bits=4, act_scales=act_scales, alpha=0.5)

    torch.testing.assert_close(out, ref)


def test_awq_autograd_ste() -> None:
    """Verify AWQ straight-through estimator backward pass propagates gradients correctly."""
    w = torch.randn(8, 16, dtype=torch.float32, requires_grad=True)
    out = awq_compress(w)
    loss = (out * 3.0).sum()
    loss.backward()

    assert w.grad is not None
    assert w.grad.shape == w.shape
    # Under STE, grad_output flows directly back scaled by 3.0
    expected_grad = torch.full_like(w, 3.0)
    torch.testing.assert_close(w.grad, expected_grad)


def test_awq_amp_compliance() -> None:
    """Verify torch.amp compliance (Rule 2.1) across float16 and bfloat16, and within autocast."""
    # float16 input
    w_fp16 = torch.randn(8, 16, dtype=torch.float16)
    out_fp16 = awq_compress(w_fp16)
    assert out_fp16.dtype == torch.float16
    assert out_fp16.shape == w_fp16.shape

    # bfloat16 input
    w_bf16 = torch.randn(8, 16, dtype=torch.bfloat16)
    out_bf16 = awq_compress(w_bf16)
    assert out_bf16.dtype == torch.bfloat16
    assert out_bf16.shape == w_bf16.shape

    # torch.amp.autocast context
    with autocast("cpu", dtype=torch.bfloat16):
        w = torch.randn(8, 16)
        out = awq_compress(w)
        assert isinstance(out, torch.Tensor)
        assert out.shape == w.shape


def test_awq_group_sizes_and_dimensions() -> None:
    """Verify AWQ handling of various group sizes, 1D weights, and custom bit widths."""
    # 1D weight tensor
    w_1d = torch.randn(64, dtype=torch.float32)
    out_1d = awq_compress(w_1d, group_size=16, bits=4)
    assert out_1d.shape == w_1d.shape
    assert out_1d.dtype == w_1d.dtype

    # Custom bit widths (bits=8)
    w_8bit = torch.randn(8, 16, dtype=torch.float32)
    out_8bit = awq_compress(w_8bit, group_size=8, bits=8)
    assert out_8bit.shape == w_8bit.shape

    # group_size larger than features / non-divisible fallback
    w_odd = torch.randn(8, 15, dtype=torch.float32)
    out_odd = awq_compress(w_odd, group_size=16, bits=4)
    assert out_odd.shape == w_odd.shape


def test_awq_top_level_dispatch() -> None:
    """Verify top-level dispatch noc.awq works seamlessly."""
    w = torch.randn(8, 16, dtype=torch.float32)
    out = noc.awq(w)
    assert isinstance(out, torch.Tensor)
    assert out.shape == w.shape
    assert out.dtype == w.dtype


def test_no_forbidden_patterns() -> None:
    """Verify absence of forbidden patterns (torch.cuda.amp, raw time.time(), etc.)."""
    from pathlib import Path

    awq_dir = Path(__file__).parent.parent.parent.parent / "neuralops" / "ops" / "cuda" / "awq"
    for py_file in awq_dir.glob("*.py"):
        src = py_file.read_text()
        assert "torch.cuda.amp" not in src, f"Deprecated torch.cuda.amp found in {py_file.name}"
        assert "time.time()" not in src, f"Unsynchronized time.time() found in {py_file.name}"
