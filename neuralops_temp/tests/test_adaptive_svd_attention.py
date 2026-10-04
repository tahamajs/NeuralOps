"""Hardening test suite for adaptive_svd_attention.py (AdaptiveSVDAttention)."""
from .testing_utils import MockConfig, DummyDistilBertAttention
import pytest
import torch
import torch.nn as nn
from neuralops.attentions.adaptive_svd_attention import (AdaptiveSVDAttention,
                                                         inject_adaptive_svd)
from neuralops.core.registry import ATTENTION_REGISTRY

class DummyRobertaSelfAttention(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

class DummyRobertaAttention(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.self = DummyRobertaSelfAttention(dim)
        self.output = nn.Module()
        self.output.dense = nn.Linear(dim, dim)

class DummyGPT2Attention(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.c_attn = nn.Linear(dim, 3 * dim)
        self.c_proj = nn.Linear(dim, dim)

class DummyQwenAttention(nn.Module):
    def __init__(self, dim=64, num_kv_heads=2, head_dim=16):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, num_kv_heads * head_dim)
        self.v_proj = nn.Linear(dim, num_kv_heads * head_dim)
        self.o_proj = nn.Linear(dim, dim)

class DummyEmptyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.foo = nn.Linear(64, 64)

# --- Registry ---
def test_registry():
    assert ATTENTION_REGISTRY.get("adaptive_svd") is not None
    assert ATTENTION_REGISTRY.get("adaptive_svd_attn") is not None
    assert ATTENTION_REGISTRY.get("adaptive_svd_attention") is not None

# --- config=None ---
def test_config_none():
    attn = AdaptiveSVDAttention(config=None, hidden_size=64, num_heads=4, svd_rank=16, residual_rank=8)
    x = torch.randn(1, 8, 64)
    out, _ = attn(x)
    assert out.shape == (1, 8, 64)
    assert torch.isfinite(out).all()

# --- RoPE odd and even head_dim ---
def test_rope_even_head_dim():
    attn = AdaptiveSVDAttention(config=MockConfig())
    x = torch.randn(1, 8, 64)
    cos = torch.randn(1, 8, 16)
    sin = torch.randn(1, 8, 16)
    out, _ = attn(x, position_embeddings=(cos, sin))
    assert out.shape == (1, 8, 64)
    assert torch.isfinite(out).all()

def test_rope_odd_head_dim():
    cfg = MockConfig(hidden_size=68, num_attention_heads=4, head_dim=17)
    attn = AdaptiveSVDAttention(config=cfg)
    x = torch.randn(1, 4, 68)
    cos = torch.randn(1, 4, 17)
    sin = torch.randn(1, 4, 17)
    out, _ = attn(x, position_embeddings=(cos, sin))
    assert out.shape == (1, 4, 68)
    assert torch.isfinite(out).all()

# --- Weight copy ---
def test_distilbert_copy():
    cfg = MockConfig()
    old_attn = DummyDistilBertAttention(64)
    attn = AdaptiveSVDAttention(config=cfg, pretrained_attn=old_attn)
    assert not attn.query.U.requires_grad
    assert not attn.query.V.requires_grad
    assert attn.query.lora_A.requires_grad
    assert attn.query.lora_B.requires_grad
    out, _ = attn(torch.randn(1, 8, 64))
    assert out.shape == (1, 8, 64)

def test_roberta_copy():
    cfg = MockConfig()
    old_attn = DummyRobertaAttention(64)
    attn = AdaptiveSVDAttention(config=cfg, pretrained_attn=old_attn)
    assert attn.is_nested_encoder
    out, _ = attn(torch.randn(1, 8, 64))
    assert out.shape == (1, 8, 64)

def test_gpt2_copy():
    cfg = MockConfig(is_decoder=True, model_type="gpt2")
    old_attn = DummyGPT2Attention(64)
    attn = AdaptiveSVDAttention(config=cfg, pretrained_attn=old_attn)
    assert attn.is_decoder
    out, _ = attn(torch.randn(1, 8, 64))
    assert out.shape == (1, 8, 64)

def test_qwen_gqa():
    cfg = MockConfig(hidden_size=64, num_attention_heads=4, num_key_value_heads=2, head_dim=16, is_decoder=True, model_type="qwen2")
    old_attn = DummyQwenAttention(dim=64, num_kv_heads=2, head_dim=16)
    attn = AdaptiveSVDAttention(config=cfg, pretrained_attn=old_attn)
    assert attn.kv_repeat == 2
    out, _ = attn(torch.randn(1, 8, 64))
    assert out.shape == (1, 8, 64)

# --- Fail-loud ---
def test_fail_loud():
    attn = AdaptiveSVDAttention(config=MockConfig())
    with pytest.raises(RuntimeError, match="mandatory projections"):
        attn.copy_weights(DummyEmptyAttention())

def test_inject_none_on_failure():
    class DummyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = DummyEmptyAttention()
    assert inject_adaptive_svd(layer=DummyLayer(), model_config=MockConfig()) is None

# --- Decoder ---
def test_decoder():
    cfg = MockConfig(is_decoder=True, model_type="gpt2")
    attn = AdaptiveSVDAttention(config=cfg)
    out, _ = attn(torch.randn(1, 8, 64))
    assert out.shape == (1, 8, 64)
    assert torch.isfinite(out).all()

# --- inject ---
def test_inject_distilbert():
    class DummyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = DummyDistilBertAttention(64)
    layer = DummyLayer()
    res = inject_adaptive_svd(layer=layer, model_config=MockConfig())
    assert res is not None
    assert isinstance(layer.attention, AdaptiveSVDAttention)
    out, _ = layer.attention(torch.randn(1, 8, 64))
    assert out.shape == (1, 8, 64)

def test_inject_none():
    assert inject_adaptive_svd(layer=None, model=None) is None

# --- Stability ---
def test_forward_stable():
    attn = AdaptiveSVDAttention(config=MockConfig())
    out, _ = attn(torch.randn(2, 12, 64))
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
