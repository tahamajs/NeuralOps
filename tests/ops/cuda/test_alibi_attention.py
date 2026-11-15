# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for ALiBi Attention CUDA/C++ kernel and fallback dispatch (Wave 9 W9-K6).

Verifies:
1. Operator importability and symbol resolution from neuralops.ops.cuda.alibi_attention.
2. Execution through PyTorch reference fallback on CPU / non-CUDA environments.
3. Shape contracts, causal masking, bidirectional mode, and numerical stability.
4. Numerical parity between PyTorch reference and dispatch interface.
5. Autograd gradient propagation (q, k, v backward flow).
6. AMP (torch.amp.autocast) compatibility across half/bfloat16 precision.
7. Extreme sequence lengths, varying head counts, and head dimension thresholds (D <= 128 and D > 128).
"""

from __future__ import annotations

import math
import pytest
import torch

from neuralops.ops.cuda.alibi_attention import alibi_attention, alibi_attention_pytorch_ref
import neuralops.ops.cuda as noc


def _get_alibi_slopes(num_heads: int) -> torch.Tensor:
    """Helper to compute standard ALiBi slopes."""
    n = 2 ** math.floor(math.log2(num_heads))
    base = [2.0 ** (-8.0 / n * (i + 1)) for i in range(n)]
    if n < num_heads:
        extra = [2.0 ** (-8.0 / (2 * n) * (2 * i + 1)) for i in range(num_heads - n)]
        base.extend(extra)
    return torch.tensor(base[:num_heads], dtype=torch.float32)


def test_alibi_attention_import_and_symbols() -> None:
    """Verify top-level and package exports for alibi_attention."""
    assert callable(alibi_attention)
    assert callable(alibi_attention_pytorch_ref)
    assert hasattr(noc, "alibi_attention")
    assert callable(noc.alibi_attention)


def test_alibi_attention_shape_contract() -> None:
    """Verify output shape matches input q shape (B, H, L, D)."""
    B, H, L, D = 2, 4, 16, 32
    q = torch.randn(B, H, L, D)
    k = torch.randn(B, H, L, D)
    v = torch.randn(B, H, L, D)
    slopes = _get_alibi_slopes(H)

    out = alibi_attention(q, k, v, slopes, is_causal=False)
    assert out.shape == q.shape
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_alibi_attention_causal_vs_bidirectional() -> None:
    """Verify causal masking produces different output than bidirectional attention."""
    B, H, L, D = 2, 4, 16, 32
    torch.manual_seed(42)
    q = torch.randn(B, H, L, D)
    k = torch.randn(B, H, L, D)
    v = torch.randn(B, H, L, D)
    slopes = _get_alibi_slopes(H)

    out_causal = alibi_attention(q, k, v, slopes, is_causal=True)
    out_bidi = alibi_attention(q, k, v, slopes, is_causal=False)

    assert out_causal.shape == q.shape
    assert out_bidi.shape == q.shape
    assert not torch.allclose(out_causal, out_bidi)


def test_alibi_attention_numerical_parity() -> None:
    """Verify exact equivalence between dispatch output and reference on CPU fallback."""
    B, H, L, D = 2, 8, 32, 64
    torch.manual_seed(123)
    q = torch.randn(B, H, L, D)
    k = torch.randn(B, H, L, D)
    v = torch.randn(B, H, L, D)
    slopes = _get_alibi_slopes(H)

    for is_causal in (False, True):
        ref = alibi_attention_pytorch_ref(q, k, v, slopes, is_causal=is_causal)
        out = alibi_attention(q, k, v, slopes, is_causal=is_causal)
        assert torch.allclose(out, ref, atol=1e-6, rtol=1e-5)


def test_alibi_attention_autograd_gradients() -> None:
    """Verify backward gradient computation flows to q, k, and v."""
    B, H, L, D = 2, 4, 8, 16
    q = torch.randn(B, H, L, D, requires_grad=True)
    k = torch.randn(B, H, L, D, requires_grad=True)
    v = torch.randn(B, H, L, D, requires_grad=True)
    slopes = _get_alibi_slopes(H)

    out = alibi_attention(q, k, v, slopes, is_causal=True)
    loss = out.sum()
    loss.backward()

    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None
    assert q.grad.shape == q.shape
    assert k.grad.shape == k.shape
    assert v.grad.shape == v.shape
    assert not torch.isnan(q.grad).any()
    assert not torch.isnan(k.grad).any()
    assert not torch.isnan(v.grad).any()


def test_alibi_attention_torch_amp_compliance() -> None:
    """Verify execution under torch.amp.autocast on CPU without error."""
    B, H, L, D = 2, 4, 16, 32
    q = torch.randn(B, H, L, D)
    k = torch.randn(B, H, L, D)
    v = torch.randn(B, H, L, D)
    slopes = _get_alibi_slopes(H)

    with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = alibi_attention(q, k, v, slopes, is_causal=True)

    assert out.shape == q.shape
    assert not torch.isnan(out).any()


def test_alibi_attention_half_and_bfloat16_precision() -> None:
    """Verify operator supports float16 and bfloat16 tensors correctly."""
    B, H, L, D = 1, 2, 8, 16
    slopes = _get_alibi_slopes(H)

    for dtype in (torch.float16, torch.bfloat16):
        q = torch.randn(B, H, L, D, dtype=dtype)
        k = torch.randn(B, H, L, D, dtype=dtype)
        v = torch.randn(B, H, L, D, dtype=dtype)
        out = alibi_attention(q, k, v, slopes, is_causal=False)
        assert out.dtype == dtype
        assert out.shape == q.shape
        assert not torch.isnan(out).any()


def test_alibi_attention_large_head_dim_fallback() -> None:
    """Verify head dimension D > 128 gracefully executes via fallback path."""
    B, H, L, D = 1, 2, 8, 256
    q = torch.randn(B, H, L, D)
    k = torch.randn(B, H, L, D)
    v = torch.randn(B, H, L, D)
    slopes = _get_alibi_slopes(H)

    out = alibi_attention(q, k, v, slopes, is_causal=True)
    ref = alibi_attention_pytorch_ref(q, k, v, slopes, is_causal=True)
    assert out.shape == (B, H, L, D)
    assert torch.allclose(out, ref, atol=1e-5)


def test_alibi_attention_non_power_of_two_heads() -> None:
    """Verify non-power-of-2 number of heads (e.g. 6, 12) evaluates correctly."""
    for H in (3, 6, 12):
        B, L, D = 1, 16, 32
        q = torch.randn(B, H, L, D)
        k = torch.randn(B, H, L, D)
        v = torch.randn(B, H, L, D)
        slopes = _get_alibi_slopes(H)
        out = alibi_attention(q, k, v, slopes, is_causal=False)
        assert out.shape == (B, H, L, D)
        assert not torch.isnan(out).any()
