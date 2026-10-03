import pytest
import torch
import torch.nn as nn
from framework.attentions.SOAFusedAttention import (SOAFusedAttention,
                                                    inject_soa_fused_attn)


class MockConfig:
    def __init__(
        self,
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        is_decoder=False,
        model_type="bert",
    ):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.is_decoder = is_decoder
        self.model_type = model_type


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


def test_soa_fused_init_and_forward_basic():
    cfg = MockConfig()
    attn = SOAFusedAttention(config=cfg, k_rank=16)
    x = torch.randn(2, 8, 64)
    out, weights = attn(x, output_attentions=True)
    assert out.shape == (2, 8, 64)
    assert weights.shape == (2, 4, 8, 8)


def test_soa_fused_none_config():
    attn = SOAFusedAttention(config=None, hidden_size=64, num_heads=4, k_rank=16)
    x = torch.randn(2, 8, 64)
    out, _ = attn(x)
    assert out.shape == (2, 8, 64)


def test_soa_fused_distilbert_copy_and_backward():
    cfg = MockConfig()
    old_attn = DummyDistilBertAttention(64)
    attn = SOAFusedAttention(config=cfg, k_rank=16, pretrained_attn=old_attn)
    assert not attn.is_nested_encoder
    assert not attn.q_down.weight.requires_grad
    assert not attn.q_up.weight.requires_grad
    assert attn.q_up.lora_A.requires_grad
    assert attn.q_up.lora_B.requires_grad
    assert not attn.out_proj.weight.requires_grad
    assert attn.out_proj.lora_A.requires_grad

    x = torch.randn(2, 8, 64)
    out, _ = attn(x)
    loss = out.sum()
    loss.backward()
    assert attn.q_up.lora_A.grad is not None
    assert attn.out_proj.lora_A.grad is not None


def test_soa_fused_roberta_nested_backward():
    cfg = MockConfig(model_type="roberta")
    old_attn = DummyRobertaAttention(64)
    attn = SOAFusedAttention(config=cfg, k_rank=16, pretrained_attn=old_attn)
    assert attn.is_nested_encoder

    # Nested encoder bypasses out_proj, but q_up, k_up, v_up MUST have trainable grads
    x = torch.randn(2, 8, 64)
    out, _ = attn(x)
    loss = out.sum()
    loss.backward()
    assert attn.q_up.lora_A.grad is not None
    assert attn.k_up.lora_A.grad is not None
    assert attn.v_up.lora_A.grad is not None


def test_soa_fused_gpt2_copy_and_causal():
    cfg = MockConfig(is_decoder=True, model_type="gpt2")
    old_attn = DummyGPT2Attention(64)
    attn = SOAFusedAttention(config=cfg, k_rank=16, pretrained_attn=old_attn)
    assert attn.is_decoder
    x = torch.randn(2, 8, 64)
    out, _ = attn(x)
    assert out.shape == (2, 8, 64)


def test_soa_fused_qwen_gqa_and_rope():
    cfg = MockConfig(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        is_decoder=True,
        model_type="qwen2",
    )
    old_attn = DummyQwenAttention(dim=64, num_kv_heads=2, head_dim=16)
    attn = SOAFusedAttention(config=cfg, k_rank=16, pretrained_attn=old_attn)
    assert attn.num_heads == 4
    assert attn.num_kv_heads == 2

    x = torch.randn(2, 8, 64)
    cos = torch.randn(1, 8, 16)
    sin = torch.randn(1, 8, 16)
    out, _ = attn(x, position_embeddings=(cos, sin))
    assert out.shape == (2, 8, 64)


def test_soa_fused_inject_layer():
    class DummyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = DummyDistilBertAttention(64)
            self.config = MockConfig()

    lyr = DummyLayer()
    injected = inject_soa_fused_attn(layer=lyr, k_rank=16)
    assert injected is not None
    assert isinstance(lyr.attention, SOAFusedAttention)
