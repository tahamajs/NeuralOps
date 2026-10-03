import pytest
import torch
import torch.nn as nn
from framework.attentions.leka_v3 import LEKAv3Attention, inject_leka_v3


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


def test_leka_v3_basic():
    cfg = MockConfig()
    attn = LEKAv3Attention(config=cfg, hidden_size=64, num_heads=4)
    x = torch.randn(2, 8, 64)
    out, _ = attn(x)
    assert out.shape == (2, 8, 64)


def test_leka_v3_lora_trainable():
    cfg = MockConfig(lora_rank=8)
    attn = LEKAv3Attention(config=cfg, hidden_size=64, num_heads=4)
    assert attn.query.weight.requires_grad is False
    assert attn.query.lora_A.requires_grad is True
    assert attn.query.lora_B.requires_grad is True


def test_leka_v3_distilbert_weight_copy():
    cfg = MockConfig()
    old_attn = DummyDistilBertAttention(64)
    attn = LEKAv3Attention(config=cfg, hidden_size=64, num_heads=4, old_attn=old_attn)
    x = torch.randn(2, 8, 64)
    out, _ = attn(x)
    assert out.shape == (2, 8, 64)


def test_leka_v3_roberta_weight_copy():
    cfg = MockConfig()
    old_attn = DummyRobertaAttention(64)
    attn = LEKAv3Attention(config=cfg, hidden_size=64, num_heads=4, old_attn=old_attn)
    assert attn.is_nested_encoder
    x = torch.randn(2, 8, 64)
    out, _ = attn(x)
    assert out.shape == (2, 8, 64)


def test_leka_v3_gpt2_weight_copy():
    cfg = MockConfig(is_decoder=True, model_type="gpt2")
    old_attn = DummyGPT2Attention(64)
    attn = LEKAv3Attention(config=cfg, hidden_size=64, num_heads=4, old_attn=old_attn)
    assert attn.is_decoder
    x = torch.randn(2, 8, 64)
    out, _ = attn(x)
    assert out.shape == (2, 8, 64)


def test_leka_v3_qwen_gqa_and_rope():
    cfg = MockConfig(hidden_size=64, num_attention_heads=4, num_key_value_heads=2, head_dim=16, is_decoder=True, model_type="qwen2")
    old_attn = DummyQwenAttention(dim=64, num_kv_heads=2, head_dim=16)
    attn = LEKAv3Attention(config=cfg, hidden_size=64, num_heads=4, old_attn=old_attn)
    assert attn.kv_repeat == 2
    x = torch.randn(2, 8, 64)
    cos = torch.randn(2, 8, 16)
    sin = torch.randn(2, 8, 16)
    out, _ = attn(x, position_embeddings=(cos, sin))
    assert out.shape == (2, 8, 64)


def test_leka_v3_inject_safe():
    class DummyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = DummyDistilBertAttention(64)

    layer = DummyLayer()
    res = inject_leka_v3(layer=layer, model_config=MockConfig())
    assert res is not None
    x = torch.randn(2, 8, 64)
    out, _ = layer.attention(x)
    assert out.shape == (2, 8, 64)


def test_leka_v3_no_output_scale():
    cfg = MockConfig()
    attn = LEKAv3Attention(config=cfg, hidden_size=64, num_heads=4)
    for name, param in attn.named_parameters():
        assert "output_scale" not in name, f"Found output_scale parameter: {name}"
    assert not hasattr(attn.query, "output_scale")
    assert not hasattr(attn.key, "output_scale")
    assert not hasattr(attn.value, "output_scale")
    assert not hasattr(attn.out_proj, "output_scale")


def test_leka_v3_landmark_scorer_uses_k_features():
    cfg = MockConfig()
    attn = LEKAv3Attention(config=cfg, hidden_size=64, num_heads=4)
    # landmark_scorer in_features should equal 2 * feature_dim
    assert attn.landmark_scorer[0].in_features == 2 * attn.feature_dim
    # Test scoring with K_features
    B, H, N = 2, 4, 16
    K_features = torch.randn(B, H, N, 2 * attn.feature_dim)
    V = torch.randn(B, H, N, attn.head_dim)
    K_land, V_land = attn.select_landmarks(K_features, V)
    assert K_land.shape[0] == B
    assert K_land.shape[1] == H
    assert V_land.shape[0] == B
    assert V_land.shape[1] == H


def test_leka_v3_gate_bias_and_hyperparameters():
    cfg = MockConfig()
    attn = LEKAv3Attention(config=cfg, hidden_size=64, num_heads=4)
    # Gate bias initialized to +2.0
    assert torch.allclose(attn.gate, torch.full((4,), 2.0))
    # Sigmoid of gate should be > 0.85
    gate_val = torch.sigmoid(attn.gate)
    assert (gate_val > 0.85).all()
    # Num landmarks default >= 64
    assert attn.num_landmarks >= 64
    # Local window size default >= 16
    assert attn.local_window_size >= 16


def test_leka_v3_nested_encoder_injection_reset():
    class DummyNestedRobertaLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = DummyRobertaAttention(64)

    class DummyFlatLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = DummyDistilBertAttention(64)

    # 1. Nested injection must set is_nested_encoder = True
    nested_layer = DummyNestedRobertaLayer()
    injected_nested = inject_leka_v3(layer=nested_layer, model_config=MockConfig())
    assert injected_nested is not None
    assert injected_nested.is_nested_encoder is True

    # 2. Flat injection must set is_nested_encoder = False
    flat_layer = DummyFlatLayer()
    injected_flat = inject_leka_v3(layer=flat_layer, model_config=MockConfig())
    assert injected_flat is not None
    assert injected_flat.is_nested_encoder is False


def test_leka_v3_qwen_head_dim_config():
    cfg = MockConfig(hidden_size=1024, num_attention_heads=16, num_key_value_heads=8, head_dim=256)
    attn = LEKAv3Attention(config=cfg)
    assert attn.head_dim == 256
    assert attn.all_head_size == 16 * 256
    assert attn.all_kv_head_size == 8 * 256

