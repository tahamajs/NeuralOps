"""Tests for CrossAttention."""
import torch
import pytest
from neuralops.attentions.modern.cross_attention import CrossAttention
from neuralops.core.registry import ATTENTION_REGISTRY


def test_cross_attention_registry():
    assert "cross_attention" in ATTENTION_REGISTRY


def test_cross_attention_forward():
    mod = CrossAttention(embed_dim=64, num_heads=4)
    q = torch.randn(2, 8, 64)
    kv = torch.randn(2, 16, 64)
    out, weights = mod(q, context=kv)
    assert out.shape == (2, 8, 64)
    assert weights is None


def test_cross_attention_self_fallback():
    mod = CrossAttention(embed_dim=64, num_heads=4)
    q = torch.randn(2, 8, 64)
    out, _ = mod(q)
    assert out.shape == (2, 8, 64)


def test_cross_attention_masked():
    mod = CrossAttention(embed_dim=64, num_heads=4)
    q = torch.randn(2, 8, 64)
    kv = torch.randn(2, 16, 64)
    mask = torch.zeros(2, 1, 8, 16)
    out, _ = mod(q, context=kv, attention_mask=mask)
    assert out.shape == (2, 8, 64)
