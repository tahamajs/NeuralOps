"""Tests for MultiQueryAttention."""
import torch
import pytest
from neuralops.attentions.modern.mqa import MultiQueryAttention
from neuralops.core.registry import ATTENTION_REGISTRY


def test_mqa_registry():
    assert "mqa_attention" in ATTENTION_REGISTRY


def test_mqa_forward():
    mod = MultiQueryAttention(embed_dim=64, num_heads=4)
    x = torch.randn(2, 8, 64)
    out, weights = mod(x)
    assert out.shape == (2, 8, 64)
    assert weights is None


def test_mqa_causal():
    mod = MultiQueryAttention(embed_dim=64, num_heads=4, causal=True)
    x = torch.randn(2, 12, 64)
    out, _ = mod(x)
    assert out.shape == (2, 12, 64)
