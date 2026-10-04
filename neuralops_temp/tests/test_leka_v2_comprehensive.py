from .testing_utils import MockConfig, DummyDistilBertAttention
import pytest
import torch
import torch.nn as nn
from neuralops.attentions.leka_v2 import LEKAv2Attention, inject_leka_v2

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

def test_leka_v2_basic():
    cfg = MockConfig()
    attn = LEKAv2Attention(config=cfg, hidden_size=64, num_heads=4)
    x = torch.randn(2, 8, 64)
    out, _ = attn(x)
    assert out.shape == (2, 8, 64)

def test_leka_v2_lora_trainable():
    cfg = MockConfig(lora_rank=8)
    attn = LEKAv2Attention(config=cfg, hidden_size=64, num_heads=4)
    assert attn.query.weight.requires_grad is False
    assert attn.query.lora_A.requires_grad is True
    assert attn.query.lora_B.requires_grad is True

def test_leka_v2_distilbert_weight_copy():
    cfg = MockConfig()
    old_attn = DummyDistilBertAttention(64)
    attn = LEKAv2Attention(config=cfg, hidden_size=64, num_heads=4, old_attn=old_attn)
    x = torch.randn(2, 8, 64)
    out, _ = attn(x)
    assert out.shape == (2, 8, 64)

def test_leka_v2_roberta_weight_copy():
    cfg = MockConfig()
    old_attn = DummyRobertaAttention(64)
    attn = LEKAv2Attention(config=cfg, hidden_size=64, num_heads=4, old_attn=old_attn)
    assert attn.is_nested_encoder
    x = torch.randn(2, 8, 64)
    out, _ = attn(x)
    assert out.shape == (2, 8, 64)

def test_leka_v2_gpt2_weight_copy():
    cfg = MockConfig(is_decoder=True, model_type="gpt2")
    old_attn = DummyGPT2Attention(64)
    attn = LEKAv2Attention(config=cfg, hidden_size=64, num_heads=4, old_attn=old_attn)
    assert attn.is_decoder
    x = torch.randn(2, 8, 64)
    out, _ = attn(x)
    assert out.shape == (2, 8, 64)

def test_leka_v2_qwen_gqa_and_rope():
    cfg = MockConfig(hidden_size=64, num_attention_heads=4, num_key_value_heads=2, head_dim=16, is_decoder=True, model_type="qwen2")
    old_attn = DummyQwenAttention(dim=64, num_kv_heads=2, head_dim=16)
    attn = LEKAv2Attention(config=cfg, hidden_size=64, num_heads=4, old_attn=old_attn)
    assert attn.kv_repeat == 2
    x = torch.randn(2, 8, 64)
    cos = torch.randn(2, 8, 16)
    sin = torch.randn(2, 8, 16)
    out, _ = attn(x, position_embeddings=(cos, sin))
    assert out.shape == (2, 8, 64)

def test_leka_v2_inject_safe():
    class DummyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = DummyDistilBertAttention(64)

    layer = DummyLayer()
    res = inject_leka_v2(layer=layer, model_config=MockConfig())
    assert res is not None
    x = torch.randn(2, 8, 64)
    out, _ = layer.attention(x)
    assert out.shape == (2, 8, 64)

def test_leka_v2_no_output_scale():
    attn = LEKAv2Attention(hidden_size=64, num_heads=4)
    assert not hasattr(attn.query, "output_scale")
    assert not hasattr(attn.key, "output_scale")
    assert not hasattr(attn.value, "output_scale")
    assert not hasattr(attn.out_proj, "output_scale")

def test_leka_v2_adaptive_landmarks():
    attn = LEKAv2Attention(hidden_size=64, num_heads=4)
    x = torch.randn(2, 128, 64)
    out, _ = attn(x)
    assert out.shape == (2, 128, 64)

    # Test select_landmarks with explicit num_landmarks
    q = attn.query(x).view(2, 128, 4, 16).transpose(1, 2)
    k = attn.key(x).view(2, 128, 4, 16).transpose(1, 2)
    v = attn.value(x).view(2, 128, 4, 16).transpose(1, 2)
    k_feat = attn.kernel_feature_map(k)
    k_land, v_land, idx = attn.select_landmarks(k_feat, v, num_landmarks=32, return_indices=True)
    assert k_land.shape[2] == 32
    assert v_land.shape[2] == 32
    assert idx.shape[2] == 32

def test_leka_v2_softpick_numerical_stability():
    # Test softpick when all elements in a row are masked
    x = torch.full((2, 4, 16, 16), -1e4)
    p = LEKAv2Attention.softpick(x, dim=-1)
    assert not torch.isnan(p).any()
    assert not torch.isinf(p).any()
    # Should fall back to uniform distribution: sum along dim -1 should be 1.0
    assert torch.allclose(p.sum(dim=-1), torch.ones(2, 4, 16), atol=1e-5)

def test_leka_v2_gate_bias_and_hyperparameters():
    attn = LEKAv2Attention(hidden_size=64, num_heads=4)
    assert attn.landmark_ratio == 0.5
    assert attn.min_landmarks == 64
    assert attn.local_window_size == 128
    assert torch.allclose(attn.local_gate, torch.full((4,), -2.0))

def test_leka_v2_nested_encoder_injection_reset():
    class NestedLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = nn.Module()
            self.attention.self = DummyRobertaSelfAttention(64)

    lyr = NestedLayer()
    res = inject_leka_v2(layer=lyr, model_config=MockConfig())
    assert res is not None
    assert lyr.attention.self.is_nested_encoder is True

    class OuterLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = DummyDistilBertAttention(64)

    lyr_outer = OuterLayer()
    res_outer = inject_leka_v2(layer=lyr_outer, model_config=MockConfig())
    assert res_outer is not None
    assert lyr_outer.attention.is_nested_encoder is False

def test_leka_v2_qwen_head_dim_config():
    # Qwen3.5: hidden_size=1024, num_heads=8, num_kv_heads=2, head_dim=256
    cfg = MockConfig(
        hidden_size=1024,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=256,
        model_type="qwen2",
    )
    old_attn = DummyQwenAttention(dim=1024, num_kv_heads=2, head_dim=256)
    attn = LEKAv2Attention(config=cfg, hidden_size=1024, num_heads=8, old_attn=old_attn)
    assert attn.head_dim == 256
    assert attn.all_head_size == 2048
    assert attn.all_kv_head_size == 512

