from .testing_utils import MockConfig, DummyDistilBertAttention
import pytest
import torch
import torch.nn as nn
from neuralops.attentions.expert_choice_moe import (ExpertChoiceMoEAttention,
                                                    inject_expert_choice_moe)

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

def test_expert_choice_moe_init_and_forward_basic():
    cfg = MockConfig()
    attn = ExpertChoiceMoEAttention(config=cfg, n_experts=4)
    x = torch.randn(2, 8, 64)
    out, weights = attn(x, output_attentions=True)
    assert out.shape == (2, 8, 64)
    assert weights.shape == (2, 4, 8, 8)
    assert attn.aux_loss is not None

def test_expert_choice_moe_none_config():
    # Ensure config can be None without raising AttributeError
    attn = ExpertChoiceMoEAttention(config=None, hidden_size=64, num_attention_heads=4, lora_rank=4)
    x = torch.randn(2, 8, 64)
    out, _ = attn(x)
    assert out.shape == (2, 8, 64)

def test_expert_choice_moe_distilbert_weight_copy():
    cfg = MockConfig()
    old_attn = DummyDistilBertAttention(64)
    attn = ExpertChoiceMoEAttention(config=cfg, old_attn=old_attn, lora_rank=8)
    assert torch.allclose(attn.q_lin.weight, old_attn.q_lin.weight)
    assert torch.allclose(attn.out_lin.weight, old_attn.out_lin.weight)
    assert not attn.q_lin.weight.requires_grad
    assert attn.q_lin.lora_A.requires_grad

def test_expert_choice_moe_roberta_weight_copy():
    cfg = MockConfig()
    old_attn = DummyRobertaAttention(64)
    attn = ExpertChoiceMoEAttention(config=cfg, old_attn=old_attn, lora_rank=8)
    assert attn.is_nested_encoder
    assert torch.allclose(attn.q_lin.weight, old_attn.self.query.weight)
    x = torch.randn(2, 8, 64)
    out, _ = attn(x)
    assert out.shape == (2, 8, 64)

def test_expert_choice_moe_gpt2_weight_copy():
    cfg = MockConfig(is_decoder=True, model_type="gpt2")
    old_attn = DummyGPT2Attention(64)
    attn = ExpertChoiceMoEAttention(config=cfg, old_attn=old_attn, lora_rank=8)
    assert attn.is_decoder
    q_split = old_attn.c_attn.weight[:64, :]
    assert torch.allclose(attn.q_lin.weight, q_split)
    x = torch.randn(2, 8, 64)
    out, _ = attn(x)
    assert out.shape == (2, 8, 64)

def test_expert_choice_moe_qwen_gqa_and_rope():
    cfg = MockConfig(hidden_size=64, num_attention_heads=4, num_key_value_heads=2, head_dim=16, is_decoder=True, model_type="qwen2")
    old_attn = DummyQwenAttention(dim=64, num_kv_heads=2, head_dim=16)
    attn = ExpertChoiceMoEAttention(config=cfg, old_attn=old_attn)
    assert attn.kv_repeat == 2
    x = torch.randn(2, 8, 64)
    cos = torch.randn(2, 1, 8, 16)
    sin = torch.randn(2, 1, 8, 16)
    out, _ = attn(x, position_embeddings=(cos, sin))
    assert out.shape == (2, 8, 64)

def test_expert_choice_moe_inject_safe():
    class DummyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = DummyDistilBertAttention(64)

    layer = DummyLayer()
    res = inject_expert_choice_moe(layer=layer, model_config=MockConfig())
    assert res is not None
    assert isinstance(layer.attention, ExpertChoiceMoEAttention)
    x = torch.randn(2, 8, 64)
    out, _ = layer.attention(x)
    assert out.shape == (2, 8, 64)
