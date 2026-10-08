"""Tests for FlashAttentionFunction autograd."""
import torch
from neuralops.core.autograd.flash_attention import FlashAttentionFunction, flash_attention_forward_backward


def test_flash_attention_function_gradcheck():
    q = torch.randn(2, 2, 4, 8, dtype=torch.float64, requires_grad=True)
    k = torch.randn(2, 2, 4, 8, dtype=torch.float64, requires_grad=True)
    v = torch.randn(2, 2, 4, 8, dtype=torch.float64, requires_grad=True)

    out = flash_attention_forward_backward(q, k, v)
    loss = out.sum()
    loss.backward()

    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None
    assert q.grad.shape == q.shape
    assert k.grad.shape == k.shape
    assert v.grad.shape == v.shape


def test_flash_attention_function_causal():
    q = torch.randn(2, 2, 6, 8, dtype=torch.float32, requires_grad=True)
    k = torch.randn(2, 2, 6, 8, dtype=torch.float32, requires_grad=True)
    v = torch.randn(2, 2, 6, 8, dtype=torch.float32, requires_grad=True)

    out = flash_attention_forward_backward(q, k, v, causal=True)
    loss = (out ** 2).sum()
    loss.backward()

    assert not torch.isnan(q.grad).any()
