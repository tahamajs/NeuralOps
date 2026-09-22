"""Hardening test suite for aegis_spectron_fp8.py (AegisSpectronFP8Attention)."""
import pytest
import torch
import torch.nn as nn

from framework.attentions.aegis_spectron_fp8 import (
    AegisSpectronFP8Attention,
    inject_aegis_spectron_fp8,
)
from framework.core.registry import ATTENTION_REGISTRY


class MockConfig:
    def __init__(
        self,
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        is_decoder=False,
        model_type="bert",
        rank=16,
        lora_rank=8,
    ):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.is_decoder = is_decoder
        self.model_type = model_type
        self.rank = rank
        self.svd_rank = rank
        self.lora_rank = lora_rank


class DummyDistilBertAttention(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.q_lin = nn.Linear(dim, dim)
        self.k_lin = nn.Linear(dim, dim)
        self.v_lin = nn.Linear(dim, dim)
        self.out_lin = nn.Linear(dim, dim)


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
    assert ATTENTION_REGISTRY.get("aegis_spectron_fp8") is not None
    assert ATTENTION_REGISTRY.get("aegis_spectron_fp8_attn") is not None


# --- config=None ---
def test_config_none():
    attn = AegisSpectronFP8Attention(config=None, hidden_size=64, num_heads=4, rank=16, lora_rank=8)
    x = torch.randn(1, 8, 64)
    out, _ = attn(x)
    assert out.shape == (1, 8, 64)
    assert torch.isfinite(out).all()


# --- RoPE with odd/even head_dim ---
def test_rope_even_head_dim():
    attn = AegisSpectronFP8Attention(config=MockConfig())
    x = torch.randn(1, 8, 64)
    cos = torch.randn(1, 8, 16)
    sin = torch.randn(1, 8, 16)
    out, _ = attn(x, position_embeddings=(cos, sin))
    assert out.shape == (1, 8, 64)
    assert torch.isfinite(out).all()


def test_rope_odd_head_dim():
    cfg = MockConfig(hidden_size=68, num_attention_heads=4, head_dim=17)
    attn = AegisSpectronFP8Attention(config=cfg)
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
    attn = AegisSpectronFP8Attention(config=cfg, pretrained_attn=old_attn)
    assert not attn.q_proj.lin1.weight.requires_grad
    assert not attn.q_proj.lin2.weight.requires_grad
    assert attn.q_proj.lora_A.requires_grad
    assert attn.q_proj.lora_B.requires_grad
    out, _ = attn(torch.randn(1, 8, 64))
    assert out.shape == (1, 8, 64)


def test_roberta_copy():
    cfg = MockConfig()
    old_attn = DummyRobertaAttention(64)
    attn = AegisSpectronFP8Attention(config=cfg, pretrained_attn=old_attn)
    assert attn.is_nested_encoder
    out, _ = attn(torch.randn(1, 8, 64))
    assert out.shape == (1, 8, 64)


def test_gpt2_copy():
    cfg = MockConfig(is_decoder=True, model_type="gpt2")
    old_attn = DummyGPT2Attention(64)
    attn = AegisSpectronFP8Attention(config=cfg, pretrained_attn=old_attn)
    assert attn.is_decoder
    out, _ = attn(torch.randn(1, 8, 64))
    assert out.shape == (1, 8, 64)


def test_qwen_gqa():
    cfg = MockConfig(hidden_size=64, num_attention_heads=4, num_key_value_heads=2, head_dim=16, is_decoder=True, model_type="qwen2")
    old_attn = DummyQwenAttention(dim=64, num_kv_heads=2, head_dim=16)
    attn = AegisSpectronFP8Attention(config=cfg, pretrained_attn=old_attn)
    assert attn.kv_repeat == 2
    out, _ = attn(torch.randn(1, 8, 64))
    assert out.shape == (1, 8, 64)


# --- Fail-loud ---
def test_fail_loud():
    attn = AegisSpectronFP8Attention(config=MockConfig())
    with pytest.raises(RuntimeError, match="mandatory projections"):
        attn.copy_weights(DummyEmptyAttention())


def test_inject_none_on_failure():
    class DummyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = DummyEmptyAttention()
    assert inject_aegis_spectron_fp8(layer=DummyLayer(), model_config=MockConfig()) is None


# --- Decoder ---
def test_decoder():
    cfg = MockConfig(is_decoder=True, model_type="gpt2")
    attn = AegisSpectronFP8Attention(config=cfg)
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
    res = inject_aegis_spectron_fp8(layer=layer, model_config=MockConfig())
    assert res is not None
    assert isinstance(layer.attention, AegisSpectronFP8Attention)
    out, _ = layer.attention(torch.randn(1, 8, 64))
    assert out.shape == (1, 8, 64)


def test_inject_none():
    assert inject_aegis_spectron_fp8(layer=None, model=None) is None


# --- Stability ---
def test_forward_stable():
    attn = AegisSpectronFP8Attention(config=MockConfig())
    out, _ = attn(torch.randn(2, 12, 64))
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
