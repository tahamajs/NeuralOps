import pytest
import torch
import torch.nn as nn
from framework.attentions.mamba import Mamba
from framework.attentions.mamba_inject import inject_mamba


class MockConfig:
    def __init__(self, **kwargs):
        self.hidden_size = kwargs.get("hidden_size", 64)
        self.num_attention_heads = kwargs.get("num_attention_heads", 4)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)
        self.head_dim = kwargs.get("head_dim", 16)
        self.dropout = kwargs.get("dropout", 0.0)
        self.is_decoder = kwargs.get("is_decoder", False)
        self.model_type = kwargs.get("model_type", "bert")
        self.lora_rank = kwargs.get("lora_rank", 16)
        self.lora_alpha = kwargs.get("lora_alpha", 32.0)
        for k, v in kwargs.items():
            setattr(self, k, v)


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


def test_mamba_inject_basic():
    class DummyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = DummyDistilBertAttention(64)

    layer = DummyLayer()
    res = inject_mamba(layer=layer, model_config=MockConfig(), old_attn=layer.attention)
    assert res is not None
    assert isinstance(res, Mamba)
    x = torch.randn(2, 8, 64)
    out, _ = layer.attention(x)
    assert out.shape == (2, 8, 64)


def test_mamba_inject_lora_trainable():
    cfg = MockConfig(lora_rank=8)
    m = Mamba(d_model=64, config=cfg, lora_rank=8)
    assert m.in_proj.weight.requires_grad is False
    assert m.in_proj.lora_A.requires_grad is True
    assert m.in_proj.lora_B.requires_grad is True


def test_mamba_inject_distilbert_copy():
    class DummyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = DummyDistilBertAttention(64)

    layer = DummyLayer()
    res = inject_mamba(layer=layer, model_config=MockConfig(), old_attn=layer.attention)
    assert res is not None
    x = torch.randn(2, 8, 64)
    out, _ = layer.attention(x)
    assert out.shape == (2, 8, 64)


def test_mamba_inject_roberta_copy():
    class DummyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = DummyRobertaAttention(64)

    layer = DummyLayer()
    res = inject_mamba(layer=layer, model_config=MockConfig(), old_attn=layer.attention)
    assert res is not None
    x = torch.randn(2, 8, 64)
    out, _ = layer.attention(x)
    assert out.shape == (2, 8, 64)


def test_mamba_inject_gpt2_copy():
    class DummyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = DummyGPT2Attention(64)

    layer = DummyLayer()
    res = inject_mamba(layer=layer, model_config=MockConfig(is_decoder=True, model_type="gpt2"), old_attn=layer.attention)
    assert res is not None
    x = torch.randn(2, 8, 64)
    out, _ = layer.attention(x)
    assert out.shape == (2, 8, 64)


def test_mamba_inject_qwen_copy():
    class DummyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = DummyQwenAttention(64)

    layer = DummyLayer()
    res = inject_mamba(layer=layer, model_config=MockConfig(is_decoder=True, model_type="qwen2"), old_attn=layer.attention)
    assert res is not None
    x = torch.randn(2, 8, 64)
    out, _ = layer.attention(x)
    assert out.shape == (2, 8, 64)


def test_mamba_inject_safe_fallback():
    class DummyLayer(nn.Module):
        pass

    layer = DummyLayer()
    res = inject_mamba(layer=layer, model_config=None)
    assert res is not None
    assert hasattr(layer, "attention")
