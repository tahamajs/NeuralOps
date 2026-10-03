from .testing_utils import MockConfig, DummyDistilBertAttention
import pytest
import torch
import torch.nn as nn
from neuralops.attentions.hgrn2_layer import HGRN2Layer, inject_hgrn2_layer

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

def test_hgrn2_layer_basic():
    cfg = MockConfig()
    layer = HGRN2Layer(config=cfg, d_model=64)
    x = torch.randn(2, 8, 64)
    out, _ = layer(x)
    assert out.shape == (2, 8, 64)

def test_hgrn2_layer_lora_trainable():
    cfg = MockConfig(lora_rank=8)
    layer = HGRN2Layer(config=cfg, d_model=64)
    assert layer.v_proj.weight.requires_grad is False
    assert layer.v_lora_A.requires_grad is True
    assert layer.v_lora_B.requires_grad is True

def test_hgrn2_layer_distilbert_weight_copy():
    cfg = MockConfig()
    old_attn = DummyDistilBertAttention(64)
    layer = HGRN2Layer(config=cfg, d_model=64, old_attn=old_attn)
    x = torch.randn(2, 8, 64)
    out, _ = layer(x)
    assert out.shape == (2, 8, 64)

def test_hgrn2_layer_roberta_weight_copy():
    cfg = MockConfig()
    old_attn = DummyRobertaAttention(64)
    layer = HGRN2Layer(config=cfg, d_model=64, old_attn=old_attn)
    assert layer.is_nested_encoder
    x = torch.randn(2, 8, 64)
    out, _ = layer(x)
    assert out.shape == (2, 8, 64)

def test_hgrn2_layer_gpt2_weight_copy():
    cfg = MockConfig(is_decoder=True, model_type="gpt2")
    old_attn = DummyGPT2Attention(64)
    layer = HGRN2Layer(config=cfg, d_model=64, old_attn=old_attn)
    assert layer.is_decoder
    x = torch.randn(2, 8, 64)
    out, _ = layer(x)
    assert out.shape == (2, 8, 64)

def test_hgrn2_layer_qwen_gqa_and_rope():
    cfg = MockConfig(hidden_size=64, num_attention_heads=4, num_key_value_heads=2, head_dim=16, is_decoder=True, model_type="qwen2")
    old_attn = DummyQwenAttention(dim=64, num_kv_heads=2, head_dim=16)
    layer = HGRN2Layer(config=cfg, d_model=64, old_attn=old_attn)
    x = torch.randn(2, 8, 64)
    cos = torch.randn(2, 8, 16)
    sin = torch.randn(2, 8, 16)
    out, _ = layer(x, position_embeddings=(cos, sin))
    assert out.shape == (2, 8, 64)

def test_hgrn2_layer_inject_safe():
    class DummyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = DummyDistilBertAttention(64)

    layer = DummyLayer()
    res = inject_hgrn2_layer(layer=layer, model_config=MockConfig())
    assert res is not None
    x = torch.randn(2, 8, 64)
    out, _ = layer.attention(x)
    assert out.shape == (2, 8, 64)

def test_hgrn2_layer_gates_trainable_after_copy():
    cfg = MockConfig()
    old_attn = DummyDistilBertAttention(64)
    layer = HGRN2Layer(config=cfg, d_model=64, old_attn=old_attn, freeze_base=True)
    assert layer.forget_gate.weight.requires_grad is True
    assert layer.forget_gate.bias.requires_grad is True
    assert layer.input_gate.weight.requires_grad is True
    assert layer.input_gate.bias.requires_grad is True
    assert layer.g_proj.weight.requires_grad is True
    assert layer.v_proj.weight.requires_grad is False
    assert layer.out_proj.weight.requires_grad is False

def test_hgrn2_layer_backward_nested_and_decoder():
    # Test nested encoder (RoBERTa)
    cfg_nested = MockConfig(model_type="roberta")
    old_roberta = DummyRobertaAttention(64)
    layer_nested = HGRN2Layer(config=cfg_nested, d_model=64, old_attn=old_roberta)
    x = torch.randn(2, 8, 64, requires_grad=True)
    out, _ = layer_nested(x)
    loss = out.sum()
    loss.backward()
    assert layer_nested.forget_gate.weight.grad is not None
    assert layer_nested.v_lora_A.grad is not None
    assert layer_nested.out_lora_A.grad is not None

    # Test decoder (GPT-2)
    cfg_dec = MockConfig(is_decoder=True, model_type="gpt2")
    old_gpt = DummyGPT2Attention(64)
    layer_dec = HGRN2Layer(config=cfg_dec, d_model=64, old_attn=old_gpt)
    x = torch.randn(2, 8, 64, requires_grad=True)
    out, _ = layer_dec(x)
    loss = out.sum()
    loss.backward()
    assert layer_dec.forget_gate.weight.grad is not None
    assert layer_dec.v_lora_A.grad is not None
    assert layer_dec.out_lora_A.grad is not None
