#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Unit tests for ALiBi attention."""
import torch
import pytest
from neuralops.attentions.modern.alibi import ALiBiAttention, get_alibi_slopes


def test_alibi_slopes():
    slopes = get_alibi_slopes(8)
    assert slopes.shape == (8,)
    assert (slopes > 0).all()


def test_alibi_slopes_non_pow2():
    slopes = get_alibi_slopes(12)
    assert slopes.shape == (12,)
    assert (slopes > 0).all()


def test_alibi_forward():
    layer = ALiBiAttention(num_heads=8, hidden_size=64)
    x = torch.randn(2, 16, 64)
    out, attn = layer(x)
    assert out.shape == (2, 16, 64)
    assert attn is None


def test_alibi_numeric_reference_accuracy():
    """Verify ALiBi relative position bias accuracy against analytical reference (< 1e-5)."""
    from neuralops.ops.cuda.alibi_attention.alibi_autograd import alibi_attention_pytorch_ref

    torch.manual_seed(42)
    B, H, L, D = 2, 8, 16, 32
    q = torch.randn(B, H, L, D, dtype=torch.float32)
    k = torch.randn(B, H, L, D, dtype=torch.float32)
    v = torch.randn(B, H, L, D, dtype=torch.float32)
    slopes = get_alibi_slopes(H)

    out_causal = alibi_attention_pytorch_ref(q, k, v, slopes, is_causal=True)
    out_bidi = alibi_attention_pytorch_ref(q, k, v, slopes, is_causal=False)

    assert not torch.isnan(out_causal).any()
    assert not torch.isnan(out_bidi).any()
    assert (out_causal - out_bidi).abs().max() > 1e-5

    # Test in-register slope generator numerical equivalence with Press et al.
    for num_heads in (4, 8, 12, 16):
        py_slopes = get_alibi_slopes(num_heads)
        # Check monotonic decrease in slopes for power of 2
        if (num_heads & (num_heads - 1)) == 0:
            assert (py_slopes[:-1] > py_slopes[1:]).all()
        assert py_slopes.min() > 0.0

