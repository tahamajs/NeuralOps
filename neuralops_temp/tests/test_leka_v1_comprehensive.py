from .testing_utils import MockConfig, DummyDistilBertAttention
import pytest
import torch
import torch.nn as nn
from neuralops.attentions.leka_v1 import LEKAV1Attention, inject_leka_v1

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

def test_leka_v1_basic():
    cfg = MockConfig()
    attn = LEKAV1Attention(config=cfg, hidden_size=64, num_heads=4)
    x = torch.randn(2, 8, 64)
    out, _ = attn(x)
    assert out.shape == (2, 8, 64)

def test_leka_v1_lora_trainable():
    cfg = MockConfig(lora_rank=8)
    attn = LEKAV1Attention(config=cfg, hidden_size=64, num_heads=4)
    assert attn.query.weight.requires_grad is False
    assert attn.query.lora_A.requires_grad is True
    assert attn.query.lora_B.requires_grad is True

def test_leka_v1_distilbert_weight_copy():
    cfg = MockConfig()
    old_attn = DummyDistilBertAttention(64)
    attn = LEKAV1Attention(config=cfg, hidden_size=64, num_heads=4, old_attn=old_attn)
    x = torch.randn(2, 8, 64)
    out, _ = attn(x)
    assert out.shape == (2, 8, 64)

def test_leka_v1_roberta_weight_copy():
    cfg = MockConfig()
    old_attn = DummyRobertaAttention(64)
    attn = LEKAV1Attention(config=cfg, hidden_size=64, num_heads=4, old_attn=old_attn)
    assert attn.is_nested_encoder
    x = torch.randn(2, 8, 64)
    out, _ = attn(x)
    assert out.shape == (2, 8, 64)

def test_leka_v1_gpt2_weight_copy():
    cfg = MockConfig(is_decoder=True, model_type="gpt2")
    old_attn = DummyGPT2Attention(64)
    attn = LEKAV1Attention(config=cfg, hidden_size=64, num_heads=4, old_attn=old_attn)
    assert attn.is_decoder
    x = torch.randn(2, 8, 64)
    out, _ = attn(x)
    assert out.shape == (2, 8, 64)

def test_leka_v1_qwen_gqa_and_rope():
    cfg = MockConfig(hidden_size=64, num_attention_heads=4, num_key_value_heads=2, head_dim=16, is_decoder=True, model_type="qwen2")
    old_attn = DummyQwenAttention(dim=64, num_kv_heads=2, head_dim=16)
    attn = LEKAV1Attention(config=cfg, hidden_size=64, num_heads=4, old_attn=old_attn)
    assert attn.kv_repeat == 2
    x = torch.randn(2, 8, 64)
    cos = torch.randn(2, 8, 16)
    sin = torch.randn(2, 8, 16)
    out, _ = attn(x, position_embeddings=(cos, sin))
    assert out.shape == (2, 8, 64)

def test_leka_v1_inject_safe():
    class DummyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = DummyDistilBertAttention(64)

    layer = DummyLayer()
    res = inject_leka_v1(layer=layer, model_config=MockConfig())
    assert res is not None
    x = torch.randn(2, 8, 64)
    out, _ = layer.attention(x)
    assert out.shape == (2, 8, 64)
