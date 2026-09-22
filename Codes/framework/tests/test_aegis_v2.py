"""Hardening test suite for aegis_v2.py (AegisAttentionV2)."""
import pytest
import torch
import torch.nn as nn

from framework.attentions.aegis_v2 import (
    AegisAttentionV2,
    _rotate_half,
    inject_aegis_v2,
)
from framework.core.registry import ATTENTION_REGISTRY


class MockConfig:
    def __init__(self, hidden_size=64, num_attention_heads=4, num_key_value_heads=4,
                 head_dim=16, is_decoder=False, model_type="bert"):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.is_decoder = is_decoder
        self.model_type = model_type


class DummyDistilBertAttn(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.q_lin = nn.Linear(dim, dim)
        self.k_lin = nn.Linear(dim, dim)
        self.v_lin = nn.Linear(dim, dim)
        self.out_lin = nn.Linear(dim, dim)


class DummyRobertaSelfAttn(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)


class DummyRobertaAttn(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.self = DummyRobertaSelfAttn(dim)
        self.output = nn.Module()
        self.output.dense = nn.Linear(dim, dim)


class DummyGPT2Attn(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.c_attn = nn.Linear(dim, 3 * dim)
        self.c_proj = nn.Linear(dim, dim)


class DummyQwenAttn(nn.Module):
    def __init__(self, dim=64, num_kv_heads=2, head_dim=16):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, num_kv_heads * head_dim)
        self.v_proj = nn.Linear(dim, num_kv_heads * head_dim)
        self.o_proj = nn.Linear(dim, dim)


class DummyEmptyAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.foo = nn.Linear(64, 64)


# --- Registry ---
def test_registry():
    assert ATTENTION_REGISTRY.get("aegis_v2") is not None


# --- config=None ---
def test_config_none():
    attn = AegisAttentionV2(config=None, hidden_size=64, num_heads=4)
    out, _ = attn(torch.randn(1, 8, 64))
    assert out.shape == (1, 8, 64)
    assert torch.isfinite(out).all()


# --- _rotate_half ---
def test_rotate_half_even():
    x = torch.randn(2, 4, 8)
    assert _rotate_half(x).shape == x.shape


def test_rotate_half_odd():
    x = torch.randn(2, 4, 9)
    assert _rotate_half(x).shape == x.shape


def test_rotate_half_roundtrip_even():
    x = torch.randn(1, 3, 8)
    assert torch.allclose(_rotate_half(_rotate_half(x)), -x, atol=1e-5)


def test_rotate_half_roundtrip_odd():
    x = torch.randn(1, 3, 9)
    rr = _rotate_half(_rotate_half(x))
    assert torch.allclose(rr[..., :-1], -x[..., :-1], atol=1e-5)
    assert torch.allclose(rr[..., -1:], x[..., -1:], atol=1e-5)


# --- Weight copy ---
def _check_frozen(attn):
    for proj in [attn.q_proj, attn.k_proj, attn.v_proj]:
        assert proj.lora_A.requires_grad
        assert proj.lora_B.requires_grad


def test_distilbert_copy():
    attn = AegisAttentionV2(config=MockConfig(), pretrained_attn=DummyDistilBertAttn(64))
    _check_frozen(attn)
    out, _ = attn(torch.randn(1, 8, 64))
    assert out.shape == (1, 8, 64)


def test_roberta_copy():
    attn = AegisAttentionV2(config=MockConfig(), pretrained_attn=DummyRobertaAttn(64))
    assert attn.is_nested_encoder
    out, _ = attn(torch.randn(1, 8, 64))
    assert out.shape == (1, 8, 64)


def test_gpt2_copy():
    cfg = MockConfig(is_decoder=True, model_type="gpt2")
    attn = AegisAttentionV2(config=cfg, pretrained_attn=DummyGPT2Attn(64))
    assert attn.is_decoder
    out, _ = attn(torch.randn(1, 8, 64))
    assert out.shape == (1, 8, 64)


def test_qwen_gqa():
    cfg = MockConfig(hidden_size=64, num_attention_heads=4, num_key_value_heads=2,
                     head_dim=16, is_decoder=True, model_type="qwen2")
    attn = AegisAttentionV2(config=cfg, pretrained_attn=DummyQwenAttn(64, 2, 16))
    assert attn.kv_repeat == 2
    out, _ = attn(torch.randn(1, 8, 64))
    assert out.shape == (1, 8, 64)


# --- Fail-loud ---
def test_fail_loud():
    attn = AegisAttentionV2(config=MockConfig())
    with pytest.raises(RuntimeError, match="mandatory projections"):
        attn.copy_weights(DummyEmptyAttn())


def test_inject_none_on_failure():
    class DummyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = DummyEmptyAttn()
    assert inject_aegis_v2(layer=DummyLayer(), model_config=MockConfig()) is None


# --- RoPE ---
def test_rope():
    cfg = MockConfig(is_decoder=True, model_type="qwen2")
    attn = AegisAttentionV2(config=cfg, pretrained_attn=DummyQwenAttn(64, 2, 16))
    cos = torch.randn(1, 8, 16)
    sin = torch.randn(1, 8, 16)
    out, _ = attn(torch.randn(1, 8, 64), position_embeddings=(cos, sin))
    assert out.shape == (1, 8, 64)
    assert torch.isfinite(out).all()


def test_rope_odd_head_dim():
    attn = AegisAttentionV2(config=MockConfig(hidden_size=68, num_attention_heads=4, head_dim=17))
    cos = torch.randn(1, 4, 17)
    sin = torch.randn(1, 4, 17)
    out, _ = attn(torch.randn(1, 4, 68), position_embeddings=(cos, sin))
    assert out.shape == (1, 4, 68)
    assert torch.isfinite(out).all()


# --- Decoder ---
def test_decoder():
    cfg = MockConfig(is_decoder=True, model_type="gpt2")
    attn = AegisAttentionV2(config=cfg)
    out, _ = attn(torch.randn(1, 8, 64))
    assert out.shape == (1, 8, 64)
    assert torch.isfinite(out).all()


# --- inject ---
def test_inject_distilbert():
    class DummyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = DummyDistilBertAttn(64)
    layer = DummyLayer()
    res = inject_aegis_v2(layer=layer, model_config=MockConfig())
    assert res is not None
    assert isinstance(layer.attention, AegisAttentionV2)
    out, _ = layer.attention(torch.randn(1, 8, 64))
    assert out.shape == (1, 8, 64)


def test_inject_none():
    assert inject_aegis_v2(layer=None, model=None) is None


# --- Stability ---
def test_forward_stable():
    attn = AegisAttentionV2(config=MockConfig())
    out, _ = attn(torch.randn(2, 12, 64))
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
