from .testing_utils import MockConfig, DummyDistilBertAttention
import pytest
import torch
import torch.nn as nn
from neuralops.attentions.gated_deltanet_block import (
    GatedDeltaNetBlock, inject_gated_deltanet_block)

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

def test_gated_deltanet_block_basic():
    cfg = MockConfig()
    block = GatedDeltaNetBlock(config=cfg, d_model=64, num_heads=4)
    x = torch.randn(2, 8, 64)
    out, _ = block(x)
    assert out.shape == (2, 8, 64)

def test_gated_deltanet_block_lora_trainable():
    cfg = MockConfig(lora_rank=8)
    block = GatedDeltaNetBlock(config=cfg, d_model=64, num_heads=4)
    assert block.deltanet.q_proj.weight.requires_grad is False
    assert block.deltanet.q_proj.lora_A.requires_grad is True
    assert block.deltanet.q_proj.lora_B.requires_grad is True

def test_gated_deltanet_block_distilbert_weight_copy():
    cfg = MockConfig()
    old_attn = DummyDistilBertAttention(64)
    block = GatedDeltaNetBlock(config=cfg, d_model=64, num_heads=4)
    block.copy_weights(old_attn)
    x = torch.randn(2, 8, 64)
    out, _ = block(x)
    assert out.shape == (2, 8, 64)

def test_gated_deltanet_block_roberta_weight_copy():
    cfg = MockConfig()
    old_attn = DummyRobertaAttention(64)
    block = GatedDeltaNetBlock(config=cfg, d_model=64, num_heads=4)
    block.copy_weights(old_attn)
    x = torch.randn(2, 8, 64)
    out, _ = block(x)
    assert out.shape == (2, 8, 64)

def test_gated_deltanet_block_gpt2_weight_copy():
    cfg = MockConfig(is_decoder=True, model_type="gpt2")
    old_attn = DummyGPT2Attention(64)
    block = GatedDeltaNetBlock(config=cfg, d_model=64, num_heads=4)
    block.copy_weights(old_attn)
    x = torch.randn(2, 8, 64)
    out, _ = block(x)
    assert out.shape == (2, 8, 64)

def test_gated_deltanet_block_qwen_gqa_and_rope():
    cfg = MockConfig(hidden_size=64, num_attention_heads=4, num_key_value_heads=2, head_dim=16, is_decoder=True, model_type="qwen2")
    old_attn = DummyQwenAttention(dim=64, num_kv_heads=2, head_dim=16)
    block = GatedDeltaNetBlock(config=cfg, d_model=64, num_heads=4)
    block.copy_weights(old_attn)
    assert block.deltanet.kv_repeat == 2
    x = torch.randn(2, 8, 64)
    cos = torch.randn(2, 8, 16)
    sin = torch.randn(2, 8, 16)
    out, _ = block(x, position_embeddings=(cos, sin))
    assert out.shape == (2, 8, 64)

def test_gated_deltanet_block_inject_safe():
    class DummyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = DummyDistilBertAttention(64)

    layer = DummyLayer()
    res = inject_gated_deltanet_block(layer=layer, model_config=MockConfig())
    assert res is not None
    x = torch.randn(2, 8, 64)
    out, _ = layer.attention(x)
    assert out.shape == (2, 8, 64)
