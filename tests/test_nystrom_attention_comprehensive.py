from .testing_utils import MockConfig, DummyDistilBertAttention
import pytest
import torch
import torch.nn as nn
from neuralops.attentions.nystrom_attention import (NystromAttention,
                                                    inject_nystrom)

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

def test_nystrom_init_and_forward_basic():
    cfg = MockConfig()
    attn = NystromAttention(config=cfg, num_landmarks=16)
    x = torch.randn(2, 8, 64)
    out, weights = attn(x, output_attentions=True)
    assert out.shape == (2, 8, 64)

def test_nystrom_none_config():
    attn = NystromAttention(config=None, hidden_size=64, num_heads=4, lora_rank=4, num_landmarks=16)
    x = torch.randn(2, 8, 64)
    out, _ = attn(x)
    assert out.shape == (2, 8, 64)

def test_nystrom_distilbert_weight_copy():
    cfg = MockConfig()
    old_attn = DummyDistilBertAttention(64)
    attn = NystromAttention(config=cfg, old_attn=old_attn, lora_rank=8, num_landmarks=16)
    assert torch.allclose(attn.q_lin.weight, old_attn.q_lin.weight)
    assert torch.allclose(attn.out_lin.weight, old_attn.out_lin.weight)
    assert not attn.q_lin.weight.requires_grad
    assert attn.q_lin.lora_A.requires_grad

def test_nystrom_roberta_weight_copy():
    cfg = MockConfig()
    old_attn = DummyRobertaAttention(64)
    attn = NystromAttention(config=cfg, old_attn=old_attn, lora_rank=8, num_landmarks=16)
    assert attn.is_nested_encoder
    assert torch.allclose(attn.q_lin.weight, old_attn.self.query.weight)
    x = torch.randn(2, 8, 64)
    out, _ = attn(x)
    assert out.shape == (2, 8, 64)

def test_nystrom_gpt2_weight_copy():
    cfg = MockConfig(is_decoder=True, model_type="gpt2")
    old_attn = DummyGPT2Attention(64)
    attn = NystromAttention(config=cfg, old_attn=old_attn, lora_rank=8, num_landmarks=16)
    assert attn.is_decoder
    q_split = old_attn.c_attn.weight[:64, :]
    assert torch.allclose(attn.q_lin.weight, q_split)
    x = torch.randn(2, 8, 64)
    out, _ = attn(x)
    assert out.shape == (2, 8, 64)

def test_nystrom_qwen_gqa_and_rope():
    cfg = MockConfig(hidden_size=64, num_attention_heads=4, num_key_value_heads=2, head_dim=16, is_decoder=True, model_type="qwen2")
    old_attn = DummyQwenAttention(dim=64, num_kv_heads=2, head_dim=16)
    attn = NystromAttention(config=cfg, old_attn=old_attn, num_landmarks=16)
    assert attn.kv_repeat == 2
    x = torch.randn(2, 8, 64)
    cos = torch.randn(2, 1, 8, 16)
    sin = torch.randn(2, 1, 8, 16)
    out, _ = attn(x, cos=cos, sin=sin)
    assert out.shape == (2, 8, 64)

def test_nystrom_inject_safe():
    class DummyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = DummyDistilBertAttention(64)

    layer = DummyLayer()
    res = inject_nystrom(layer=layer, model_config=MockConfig())
    assert res is not None
    assert isinstance(layer.attention, NystromAttention)
    x = torch.randn(2, 8, 64)
    out, _ = layer.attention(x)
    assert out.shape == (2, 8, 64)

def test_nystrom_short_and_long_sequence():
    cfg = MockConfig(hidden_size=64, num_attention_heads=4, is_decoder=False)
    attn = NystromAttention(config=cfg, num_landmarks=16)

    # Short sequence (N=32 <= 128) -> triggers dense fallback
    x_short = torch.randn(2, 32, 64)
    out_short, _ = attn(x_short)
    assert out_short.shape == (2, 32, 64)

    # Long sequence (N=256 > 128) -> triggers landmark pooling + pinv
    x_long = torch.randn(2, 256, 64)
    out_long, _ = attn(x_long)
    assert out_long.shape == (2, 256, 64)

def test_nystrom_backward_nested_and_decoder():
    # RoBERTa nested encoder backward test
    cfg_roberta = MockConfig(hidden_size=64, num_attention_heads=4, is_decoder=False)
    old_roberta = DummyRobertaAttention(64)
    attn_roberta = NystromAttention(config=cfg_roberta, old_attn=old_roberta, lora_rank=8, num_landmarks=16)
    x_roberta = torch.randn(2, 32, 64, requires_grad=True)
    out_roberta, _ = attn_roberta(x_roberta)
    loss = out_roberta.sum()
    loss.backward()
    assert attn_roberta.query.lora_A.grad is not None
    assert attn_roberta.query.weight.grad is None  # Frozen base weight

    # GPT-2 decoder backward test
    cfg_gpt = MockConfig(hidden_size=64, num_attention_heads=4, is_decoder=True, model_type="gpt2")
    old_gpt = DummyGPT2Attention(64)
    attn_gpt = NystromAttention(config=cfg_gpt, old_attn=old_gpt, lora_rank=8, num_landmarks=16)
    x_gpt = torch.randn(2, 32, 64, requires_grad=True)
    out_gpt, _ = attn_gpt(x_gpt)
    loss_gpt = out_gpt.sum()
    loss_gpt.backward()
    assert attn_gpt.query.lora_A.grad is not None
    assert attn_gpt.out_proj.lora_A.grad is not None
