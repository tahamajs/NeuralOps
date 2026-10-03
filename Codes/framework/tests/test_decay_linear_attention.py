"""Comprehensive test suite for Decay Linear Attention (decay_linear_attention.py)."""

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.attentions.decay_linear_attention import (DecayLinearAttention,
                                                         _rotate_half,
                                                         inject_decay_linear)
from framework.core.registry import ATTENTION_REGISTRY


class MockConfig:
    def __init__(
        self,
        hidden_size: int = 64,
        num_attention_heads: int = 4,
        num_key_value_heads: int = 4,
        head_dim: int = 16,
        is_decoder: bool = False,
    ):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.is_decoder = is_decoder


def test_registry():
    assert "decay_linear_attention" in ATTENTION_REGISTRY.list_available()
    assert "decay_linear_attn" in ATTENTION_REGISTRY.list_available()
    assert ATTENTION_REGISTRY.get("decay_linear_attention") is DecayLinearAttention


def test_forward_bidirectional():
    cfg = MockConfig(is_decoder=False)
    layer = DecayLinearAttention(config=cfg, d_model=64, num_heads=4)
    x = torch.randn(2, 16, 64)
    mask = torch.ones(2, 16)
    out, attn = layer(x, attention_mask=mask)
    assert out.shape == (2, 16, 64)
    assert attn is None
    assert not torch.isnan(out).any()


def test_forward_causal():
    cfg = MockConfig(is_decoder=True)
    layer = DecayLinearAttention(config=cfg, d_model=64, num_heads=4)
    x = torch.randn(2, 16, 64)
    out, attn = layer(x)
    assert out.shape == (2, 16, 64)
    assert not torch.isnan(out).any()


def test_chunked_scan_numerical_equivalence():
    cfg = MockConfig(is_decoder=True)
    layer = DecayLinearAttention(config=cfg, d_model=64, num_heads=4)
    # L = 64 tests chunking with chunk_size=32
    B, H, L, d = 2, 4, 64, 16
    q = F.elu(torch.randn(B, H, L, d)) + 1.0
    k = F.elu(torch.randn(B, H, L, d)) + 1.0
    v = torch.randn(B, H, L, d)
    lambdas = torch.full((B, H, L), 0.95)

    out_chunked = layer._decay_scan_linear(q, k, v, lambdas, chunk_size=32)
    out_single = layer._decay_scan_linear(q, k, v, lambdas, chunk_size=128)

    assert torch.allclose(out_chunked, out_single, atol=1e-4, rtol=1e-4)


def test_rotate_half_odd_dimension():
    x = torch.randn(2, 4, 15)
    rot = _rotate_half(x)
    assert rot.shape == x.shape
    assert torch.allclose(rot[..., 14], x[..., 14])


def test_copy_weights_distilbert():
    class MockDistilBertAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_lin = nn.Linear(64, 64)
            self.k_lin = nn.Linear(64, 64)
            self.v_lin = nn.Linear(64, 64)
            self.out_lin = nn.Linear(64, 64)

    mock = MockDistilBertAttn()
    cfg = MockConfig()
    layer = DecayLinearAttention(config=cfg, d_model=64, num_heads=4)
    layer.copy_weights(mock)
    assert torch.allclose(layer.q_proj.weight, mock.q_lin.weight)
    assert torch.allclose(layer.out_proj.weight, mock.out_lin.weight)


def test_copy_weights_roberta_nested():
    class MockRobertaSelf(nn.Module):
        def __init__(self):
            super().__init__()
            self.query = nn.Linear(64, 64)
            self.key = nn.Linear(64, 64)
            self.value = nn.Linear(64, 64)

    class MockRobertaOutput(nn.Module):
        def __init__(self):
            super().__init__()
            self.dense = nn.Linear(64, 64)

    class MockRobertaAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.self = MockRobertaSelf()
            self.output = MockRobertaOutput()

    mock = MockRobertaAttention()
    cfg = MockConfig()
    layer = DecayLinearAttention(config=cfg, d_model=64, num_heads=4)
    layer.copy_weights(mock)
    assert layer.is_nested_encoder
    assert torch.allclose(layer.q_proj.weight, mock.self.query.weight)


def test_copy_weights_gpt2():
    class MockGPT2Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.c_attn = nn.Linear(64, 192)
            self.c_proj = nn.Linear(64, 64)

    mock = MockGPT2Attn()
    cfg = MockConfig(is_decoder=True)
    layer = DecayLinearAttention(config=cfg, d_model=64, num_heads=4)
    layer.copy_weights(mock)
    assert layer.q_proj.weight.shape == (64, 64)
    assert layer.out_proj.weight.shape == (64, 64)


def test_copy_weights_missing_raises():
    class IncompleteAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_lin = nn.Linear(64, 64)

    mock = IncompleteAttn()
    cfg = MockConfig()
    layer = DecayLinearAttention(config=cfg, d_model=64, num_heads=4)
    with pytest.raises(RuntimeError):
        layer.copy_weights(mock)


def test_inject_decay_linear_single_layer():
    class MockAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_lin = nn.Linear(64, 64)
            self.k_lin = nn.Linear(64, 64)
            self.v_lin = nn.Linear(64, 64)
            self.out_lin = nn.Linear(64, 64)

    class MockLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = MockAttn()

    layer = MockLayer()
    cfg = MockConfig()
    ret = inject_decay_linear(layer=layer, model_config=cfg, layer_idx=0, total_layers=6)
    assert isinstance(ret, DecayLinearAttention)
    assert layer.attention is ret


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
