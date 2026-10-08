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
    layer = ALiBiAttention(num_heads=8)
    x = torch.randn(2, 16, 64)
    out, attn = layer(x)
    assert out.shape == (2, 16, 64)
    assert attn is None
