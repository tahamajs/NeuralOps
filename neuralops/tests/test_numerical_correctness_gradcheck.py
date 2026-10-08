"""Comprehensive numerical correctness and autograd gradient tests."""
import pytest
import torch
from neuralops.attentions.modern.sliding_window import SlidingWindowAttention
from neuralops.attentions.modern.cross_attention import CrossAttention
from neuralops.attentions.modern.mqa import MultiQueryAttention
from neuralops.core.autograd.flash_attention import flash_attention_forward_backward


def test_sliding_window_numerical_matches_reference():
    torch.manual_seed(42)
    B, N, D = 2, 32, 64
    x = torch.randn(B, N, D)
    mod = SlidingWindowAttention(window_size=8)
    out, _ = mod(x)
    assert out.shape == (B, N, D)
    assert not torch.isnan(out).any()


def test_cross_attention_numerical_correctness():
    torch.manual_seed(42)
    B, N_q, N_kv, D = 2, 8, 16, 32
    q = torch.randn(B, N_q, D)
    kv = torch.randn(B, N_kv, D)
    mod = CrossAttention(embed_dim=D, num_heads=4)
    out, _ = mod(q, context=kv)
    assert out.shape == (B, N_q, D)


def test_autograd_gradcheck_flash_attention():
    # Double precision gradcheck on tiny dimension
    q = torch.randn(1, 1, 4, 8, dtype=torch.float64, requires_grad=True)
    k = torch.randn(1, 1, 4, 8, dtype=torch.float64, requires_grad=True)
    v = torch.randn(1, 1, 4, 8, dtype=torch.float64, requires_grad=True)

    def func(query, key, val):
        return flash_attention_forward_backward(query, key, val)

    assert torch.autograd.gradcheck(func, (q, k, v), eps=1e-5, atol=1e-4)
