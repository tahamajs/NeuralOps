from .testing_utils import MockConfig, DummyDistilBertAttention
import pytest
import torch
import torch.nn as nn
from neuralops.attentions.lola_attention import (LoLAAttention,
                                                 inject_lola_attention)

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

def test_lola_basic():
    cfg = MockConfig()
    attn = LoLAAttention(config=cfg, rank=16)
    x = torch.randn(2, 8, 64)
    out, _ = attn(x)
    assert out.shape == (2, 8, 64)

def test_lola_lora_trainable():
    cfg = MockConfig(lora_rank=8)
    attn = LoLAAttention(config=cfg, rank=16)
    assert attn.q_proj.U.requires_grad is False
    assert attn.q_proj.V.requires_grad is False
    assert attn.q_proj.lora_A.requires_grad is True
    assert attn.q_proj.lora_B.requires_grad is True

def test_lola_distilbert_weight_copy():
    cfg = MockConfig()
    old_attn = DummyDistilBertAttention(64)
    attn = LoLAAttention(config=cfg, rank=16, pretrained_attn=old_attn)
    x = torch.randn(2, 8, 64)
    out, _ = attn(x)
    assert out.shape == (2, 8, 64)

def test_lola_roberta_weight_copy():
    cfg = MockConfig()
    old_attn = DummyRobertaAttention(64)
    attn = LoLAAttention(config=cfg, rank=16, pretrained_attn=old_attn)
    assert attn.is_nested_encoder
    x = torch.randn(2, 8, 64)
    out, _ = attn(x)
    assert out.shape == (2, 8, 64)

def test_lola_gpt2_weight_copy():
    cfg = MockConfig(is_decoder=True, model_type="gpt2")
    old_attn = DummyGPT2Attention(64)
    attn = LoLAAttention(config=cfg, rank=16, pretrained_attn=old_attn)
    assert attn.is_decoder
    x = torch.randn(2, 8, 64)
    out, _ = attn(x)
    assert out.shape == (2, 8, 64)

def test_lola_qwen_gqa_and_rope():
    cfg = MockConfig(hidden_size=64, num_attention_heads=4, num_key_value_heads=2, head_dim=16, is_decoder=True, model_type="qwen2")
    old_attn = DummyQwenAttention(dim=64, num_kv_heads=2, head_dim=16)
    attn = LoLAAttention(config=cfg, rank=16, pretrained_attn=old_attn)
    assert attn.kv_repeat == 2
    x = torch.randn(2, 8, 64)
    cos = torch.randn(2, 8, 16)
    sin = torch.randn(2, 8, 16)
    out, _ = attn(x, position_embeddings=(cos, sin))
    assert out.shape == (2, 8, 64)

def test_lola_inject_safe():
    class DummyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = DummyDistilBertAttention(64)

    layer = DummyLayer()
    res = inject_lola_attention(layer=layer, model_config=MockConfig(), old_attn=layer.attention)
    assert res is not None
    x = torch.randn(2, 8, 64)
    out, _ = layer.attention(x)
    assert out.shape == (2, 8, 64)

def test_lola_config_none_safe():
    attn = LoLAAttention(config=None, rank=16, hidden_size=64, num_heads=4)
    assert attn.hidden_size == 64
    assert attn.num_heads == 4
    x = torch.randn(2, 8, 64)
    out, _ = attn(x)
    assert out.shape == (2, 8, 64)

def test_lola_base_layer_peft_unwrapping():
    class DummyLoRALikeWrapper(nn.Module):
        def __init__(self, in_dim, out_dim):
            super().__init__()
            self.base_layer = nn.Linear(in_dim, out_dim)

        @property
        def weight(self):
            return self.base_layer.weight

        @property
        def bias(self):
            return self.base_layer.bias

    class DummyWrappedGPT2Attention(nn.Module):
        def __init__(self, dim=64):
            super().__init__()
            self.c_attn = DummyLoRALikeWrapper(dim, 3 * dim)
            self.c_proj = DummyLoRALikeWrapper(dim, dim)

    cfg = MockConfig(is_decoder=True, model_type="gpt2")
    wrapped_attn = DummyWrappedGPT2Attention(64)
    attn = LoLAAttention(config=cfg, rank=16, pretrained_attn=wrapped_attn)
    assert attn.is_decoder
    x = torch.randn(2, 8, 64)
    out, _ = attn(x)
    assert out.shape == (2, 8, 64)

def test_lola_no_output_scale():
    cfg = MockConfig()
    attn = LoLAAttention(config=cfg, rank=16)
    assert not hasattr(attn, "output_scale")

def test_lola_qwen3_5_head_dim():
    class Qwen35Config:
        hidden_size = 1024
        num_attention_heads = 8
        num_key_value_heads = 8
        head_dim = 256
        model_type = "qwen3_5"
        is_decoder = True

    attn = LoLAAttention(config=Qwen35Config(), rank=16)
    assert attn.head_dim == 256
    assert attn.all_head_size == 2048
    assert attn.hidden_size == 1024
    assert attn.is_decoder is True

    x = torch.randn(2, 4, 1024)
    out, _ = attn(x)
    assert out.shape == (2, 4, 1024)

def test_lola_lowrank_cache_batch_and_heads():
    from neuralops.attentions.lola_attention import LowRankCache
    cache = LowRankCache(max_len=16, low_rank_k=4)
    B, H, N, D = 2, 4, 8, 16
    K = torch.randn(B, H, N, D)
    V = torch.randn(B, H, N, D)
    cache.update(K, V)
    k_cached, v_cached = cache.get_cached()
    assert k_cached is not None and v_cached is not None
    # Shape must be [B*H, C, D]
    assert k_cached.shape[0] == B * H
    assert k_cached.shape[2] == D
    assert v_cached.shape[0] == B * H
    assert v_cached.shape[2] == D

    # Test cache reset
    cache.reset()
    k_empty, v_empty = cache.get_cached()
    assert k_empty is None and v_empty is None

def test_lola_causal_mask_enforcement():
    cfg = MockConfig(is_decoder=True, model_type="gpt2")
    attn = LoLAAttention(config=cfg, rank=16)
    x = torch.randn(2, 6, 64)
    out, weights = attn(x, output_attentions=True)
    assert weights is not None
    # Future positions must have probability zero
    for i in range(6):
        for j in range(i + 1, 6):
            assert torch.all(weights[:, :, i, j] == 0.0)

def test_lola_4d_masks():
    cfg = MockConfig()
    attn = LoLAAttention(config=cfg, rank=16)
    x = torch.randn(2, 4, 64)

    # 4D boolean mask [B, 1, N, N]
    bool_mask = torch.ones(2, 1, 4, 4, dtype=torch.bool)
    bool_mask[:, :, :, 3] = False  # mask out last token
    out_bool, _ = attn(x, attention_mask=bool_mask)
    assert out_bool.shape == (2, 4, 64)

    # 4D float additive mask
    float_mask = torch.zeros(2, 1, 4, 4, dtype=torch.float32)
    float_mask[:, :, :, 3] = -10000.0
    out_float, _ = attn(x, attention_mask=float_mask)
    assert out_float.shape == (2, 4, 64)

def test_lola_nested_encoder_injection():
    class DummyBertSelf(nn.Module):
        def __init__(self):
            super().__init__()
            self.query = nn.Linear(64, 64)
            self.key = nn.Linear(64, 64)
            self.value = nn.Linear(64, 64)

    class DummyBertAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.self = DummyBertSelf()
            self.output = nn.Module()
            self.output.dense = nn.Linear(64, 64)

    class DummyBertLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = DummyBertAttention()

    class DummyLlamaLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = DummyQwenAttention(64)

    bert_layer = DummyBertLayer()
    injected_bert = inject_lola_attention(bert_layer, model_config=MockConfig(), old_attn=bert_layer.attention.self)
    assert injected_bert.is_nested_encoder is True

    llama_layer = DummyLlamaLayer()
    injected_llama = inject_lola_attention(llama_layer, model_config=MockConfig(), old_attn=llama_layer.self_attn)
    assert injected_llama.is_nested_encoder is False

