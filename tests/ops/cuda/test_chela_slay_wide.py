# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for CHELA-SLAY Wide Head Attention CUDA kernel & CPU fallback (Wave 9 / W9-K16).
"""

from __future__ import annotations

import inspect
import pytest
import torch
from torch.amp import autocast

import neuralops.ops.cuda as noc
from neuralops.ops.cuda.chela_slay_wide import (
    ChelaSlayWideFunction,
    chela_slay_wide_attention_forward,
)
from neuralops.ops.triton.chela_slay_wide_triton import _chela_slay_wide_pytorch_ref


def test_chela_slay_wide_symbol_export_and_callable() -> None:
    """Verify that symbols are properly exported from package and callable."""
    assert callable(chela_slay_wide_attention_forward)
    assert issubclass(ChelaSlayWideFunction, torch.autograd.Function)


def test_chela_slay_wide_cpu_fallback_basic() -> None:
    """Verify forward execution on CPU produces valid tensors with correct shape and dtype."""
    B, H, L, D = 2, 4, 16, 128
    q = torch.randn(B, H, L, D, dtype=torch.float32)
    k = torch.randn(B, H, L, D, dtype=torch.float32)
    v = torch.randn(B, H, L, D, dtype=torch.float32)

    out = chela_slay_wide_attention_forward(q, k, v, chunk_size=8)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (B, H, L, D)
    assert out.dtype == torch.float32
    assert out.device == q.device
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_chela_slay_wide_cpu_numerical_parity() -> None:
    """Verify output matches PyTorch reference implementation exactly on CPU."""
    torch.manual_seed(42)
    B, H, L, D = 2, 2, 12, 128
    q = torch.randn(B, H, L, D, dtype=torch.float32)
    k = torch.randn(B, H, L, D, dtype=torch.float32)
    v = torch.randn(B, H, L, D, dtype=torch.float32)

    out = chela_slay_wide_attention_forward(q, k, v, chunk_size=4)
    ref = _chela_slay_wide_pytorch_ref(q, k, v, chunk_size=4)

    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_chela_slay_wide_wide_head_dimensions() -> None:
    """Verify support across various wide head dimensions D in {64, 128, 256}."""
    for D in [64, 128, 256]:
        B, H, L = 1, 2, 8
        q = torch.randn(B, H, L, D, dtype=torch.float32)
        k = torch.randn(B, H, L, D, dtype=torch.float32)
        v = torch.randn(B, H, L, D, dtype=torch.float32)

        out = chela_slay_wide_attention_forward(q, k, v, chunk_size=4)
        assert out.shape == (B, H, L, D)
        assert not torch.isnan(out).any()


def test_chela_slay_wide_non_multiple_chunk_sizes() -> None:
    """Verify chunk padding and unpadding when sequence length L is not divisible by chunk_size."""
    B, H, L, D = 2, 2, 13, 128  # 13 is prime, not divisible by chunk_size 4 or 8
    q = torch.randn(B, H, L, D, dtype=torch.float32)
    k = torch.randn(B, H, L, D, dtype=torch.float32)
    v = torch.randn(B, H, L, D, dtype=torch.float32)

    out = chela_slay_wide_attention_forward(q, k, v, chunk_size=8)
    assert out.shape == (B, H, L, D)
    assert not torch.isnan(out).any()


def test_chela_slay_wide_autograd_backward() -> None:
    """Verify autograd backward pass computes gradients for q, k, and v."""
    torch.manual_seed(42)
    B, H, L, D = 1, 2, 8, 64
    q = torch.randn(B, H, L, D, dtype=torch.float32, requires_grad=True)
    k = torch.randn(B, H, L, D, dtype=torch.float32, requires_grad=True)
    v = torch.randn(B, H, L, D, dtype=torch.float32, requires_grad=True)

    out = chela_slay_wide_attention_forward(q, k, v, chunk_size=4)
    loss = (out * 2.0).sum()
    loss.backward()

    assert q.grad is not None and q.grad.shape == q.shape
    assert k.grad is not None and k.grad.shape == k.shape
    assert v.grad is not None and v.grad.shape == v.shape
    assert not torch.isnan(q.grad).any()
    assert not torch.isnan(k.grad).any()
    assert not torch.isnan(v.grad).any()


def test_chela_slay_wide_amp_compliance() -> None:
    """Verify torch.amp compliance (Rule 2.1) across float32, bfloat16, and float16."""
    B, H, L, D = 1, 2, 8, 128

    # Test bfloat16 tensor directly
    q_bf16 = torch.randn(B, H, L, D, dtype=torch.bfloat16)
    k_bf16 = torch.randn(B, H, L, D, dtype=torch.bfloat16)
    v_bf16 = torch.randn(B, H, L, D, dtype=torch.bfloat16)
    out_bf16 = chela_slay_wide_attention_forward(q_bf16, k_bf16, v_bf16, chunk_size=4)
    assert out_bf16.dtype == torch.bfloat16
    assert not torch.isnan(out_bf16).any()

    # Test float16 tensor directly
    q_fp16 = torch.randn(B, H, L, D, dtype=torch.float16)
    k_fp16 = torch.randn(B, H, L, D, dtype=torch.float16)
    v_fp16 = torch.randn(B, H, L, D, dtype=torch.float16)
    out_fp16 = chela_slay_wide_attention_forward(q_fp16, k_fp16, v_fp16, chunk_size=4)
    assert out_fp16.dtype == torch.float16
    assert not torch.isnan(out_fp16).any()

    # Test within torch.amp.autocast context
    with autocast("cpu", dtype=torch.bfloat16):
        q = torch.randn(B, H, L, D)
        k = torch.randn(B, H, L, D)
        v = torch.randn(B, H, L, D)
        out = chela_slay_wide_attention_forward(q, k, v, chunk_size=4)
        assert isinstance(out, torch.Tensor)
        assert not torch.isnan(out).any()


def test_chela_slay_wide_top_level_dispatch() -> None:
    """Verify top-level dispatch via neuralops.ops.cuda works seamlessly."""
    B, H, L, D = 1, 2, 8, 128
    q = torch.randn(B, H, L, D, dtype=torch.float32)
    k = torch.randn(B, H, L, D, dtype=torch.float32)
    v = torch.randn(B, H, L, D, dtype=torch.float32)

    out1 = noc.chela_slay_wide(q, k, v, chunk_size=4)
    out2 = noc.chela_slay_wide_attention_forward(q, k, v, chunk_size=4)

    assert out1.shape == (B, H, L, D)
    assert torch.allclose(out1, out2)


def test_chela_slay_wide_no_forbidden_patterns() -> None:
    """Verify absence of torch.cuda.amp, raw time.time(), etc."""
    import neuralops.ops.cuda.chela_slay_wide as pkg
    import neuralops.ops.cuda.chela_slay_wide.chela_slay_wide_autograd as autograd_mod

    for mod in [pkg, autograd_mod]:
        src = inspect.getsource(mod)
        assert "torch.cuda.amp" not in src, f"Deprecated torch.cuda.amp found in {mod.__name__}"
        assert "time.time()" not in src, f"Unsynchronized time.time() found in {mod.__name__}"
