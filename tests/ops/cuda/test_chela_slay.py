# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit and regression tests for CHELA-SLAY Attention CUDA kernel and fallbacks (Wave 9 / W9-K15).
"""

from __future__ import annotations

import inspect
import pytest
import torch
from torch.amp import autocast

import neuralops.ops.cuda as noc
from neuralops.ops.cuda.chela_slay import (
    ChelaSlayAttentionFunction,
    chela_slay_attention_forward,
)
from neuralops.ops.triton.chela_slay_triton import _chela_slay_pytorch


def test_chela_slay_cpu_forward_basic() -> None:
    """Verify CHELA-SLAY forward operator executes on CPU and preserves shape and dtype."""
    B, H, L, D = 2, 4, 32, 64
    q = torch.randn(B, H, L, D, dtype=torch.float32)
    k = torch.randn(B, H, L, D, dtype=torch.float32)
    v = torch.randn(B, H, L, D, dtype=torch.float32)

    out = chela_slay_attention_forward(q, k, v, chunk_size=16)

    assert isinstance(out, torch.Tensor)
    assert out.shape == (B, H, L, D)
    assert out.dtype == torch.float32
    assert out.device == q.device
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_chela_slay_cpu_numerical_parity() -> None:
    """Verify forward matches canonical _chela_slay_pytorch reference."""
    B, H, L, D = 2, 2, 24, 32
    q = torch.randn(B, H, L, D, dtype=torch.float32)
    k = torch.randn(B, H, L, D, dtype=torch.float32)
    v = torch.randn(B, H, L, D, dtype=torch.float32)

    out_op = chela_slay_attention_forward(q, k, v, chunk_size=8)
    out_ref = _chela_slay_pytorch(q, k, v, chunk_size=8)

    torch.testing.assert_close(out_op, out_ref, rtol=1e-5, atol=1e-5)


def test_chela_slay_function_direct_apply() -> None:
    """Verify ChelaSlayAttentionFunction.apply executes directly on CPU."""
    B, H, L, D = 1, 2, 16, 16
    q = torch.randn(B, H, L, D, dtype=torch.float32)
    k = torch.randn(B, H, L, D, dtype=torch.float32)
    v = torch.randn(B, H, L, D, dtype=torch.float32)

    out = ChelaSlayAttentionFunction.apply(q, k, v, 8)
    assert out.shape == (B, H, L, D)
    assert not torch.isnan(out).any()


def test_chela_slay_autograd_analytical_backward() -> None:
    """Verify analytical gradients are computed for q, k, v through autograd backward."""
    B, H, L, D = 2, 2, 16, 16
    q = torch.randn(B, H, L, D, dtype=torch.float32, requires_grad=True)
    k = torch.randn(B, H, L, D, dtype=torch.float32, requires_grad=True)
    v = torch.randn(B, H, L, D, dtype=torch.float32, requires_grad=True)

    out = ChelaSlayAttentionFunction.apply(q, k, v, 8)
    loss = (out**2).sum()
    loss.backward()

    assert q.grad is not None and q.grad.shape == q.shape
    assert k.grad is not None and k.grad.shape == k.shape
    assert v.grad is not None and v.grad.shape == v.shape
    assert (q.grad != 0).any().item()
    assert (k.grad != 0).any().item()
    assert (v.grad != 0).any().item()
    assert not torch.isnan(q.grad).any()
    assert not torch.isnan(k.grad).any()
    assert not torch.isnan(v.grad).any()


def test_chela_slay_amp_autocast_compliance() -> None:
    """Verify torch.amp.autocast compatibility on CPU without torch.cuda.amp."""
    B, H, L, D = 1, 2, 16, 32
    q = torch.randn(B, H, L, D, dtype=torch.bfloat16)
    k = torch.randn(B, H, L, D, dtype=torch.bfloat16)
    v = torch.randn(B, H, L, D, dtype=torch.bfloat16)

    with autocast("cpu", dtype=torch.bfloat16):
        out = chela_slay_attention_forward(q, k, v, chunk_size=8)

    assert out.shape == (B, H, L, D)
    assert out.dtype == torch.bfloat16
    assert not torch.isnan(out).any()


def test_chela_slay_top_level_noc_dispatch() -> None:
    """Verify neuralops.ops.cuda exports dispatch correctly."""
    assert hasattr(noc, "chela_slay")
    assert hasattr(noc, "chela_slay_attention_forward")

    q = torch.randn(1, 2, 16, 16)
    k = torch.randn(1, 2, 16, 16)
    v = torch.randn(1, 2, 16, 16)

    out1 = noc.chela_slay(q, k, v, chunk_size=8)
    out2 = noc.chela_slay_attention_forward(q, k, v, chunk_size=8)
    assert out1.shape == (1, 2, 16, 16)
    assert out2.shape == (1, 2, 16, 16)
    torch.testing.assert_close(out1, out2)


def test_chela_slay_arbitrary_sequence_lengths() -> None:
    """Verify handling when sequence length L is not an exact multiple of chunk_size."""
    B, H, L, D = 2, 2, 29, 32
    q = torch.randn(B, H, L, D)
    k = torch.randn(B, H, L, D)
    v = torch.randn(B, H, L, D)

    out = chela_slay_attention_forward(q, k, v, chunk_size=8)
    assert out.shape == (B, H, 29, D)
    assert not torch.isnan(out).any()


def test_no_forbidden_patterns() -> None:
    """Verify absence of torch.cuda.amp, hardcoded cuda:0, and line limit compliance."""
    from neuralops.ops.cuda.chela_slay import autograd as autograd_mod

    src = inspect.getsource(autograd_mod)
    assert "torch.cuda.amp" not in src, "Forbidden torch.cuda.amp pattern detected"
    assert "cuda:0" not in src, "Forbidden hardcoded cuda:0 pattern detected"

    lines = src.splitlines()
    assert len(lines) <= 800, f"File exceeds 800 lines limit ({len(lines)})"
