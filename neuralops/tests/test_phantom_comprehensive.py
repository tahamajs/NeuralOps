from .testing_utils import MockConfig, DummyDistilBertAttention
import pytest
import torch
import torch.nn as nn
from neuralops.attentions.phantom import (PHANTOMAttention, PHANTOMConfig,
                                          inject_phantom_attention)

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

def test_phantom_basic():
    cfg = MockConfig()
    pcfg = PHANTOMConfig(cur_rank=8, lsh_rounds=2, lsh_bucket_size=2)
    attn = PHANTOMAttention(config=cfg, phantom_cfg=pcfg)
    x = torch.randn(2, 8, 64)
    out, _ = attn(x)
    assert out.shape == (2, 8, 64)

def test_phantom_trainable_lora_weights():
    cfg = MockConfig(lora_rank=8)
    pcfg = PHANTOMConfig(cur_rank=8, lsh_rounds=2, lsh_bucket_size=2)
    attn = PHANTOMAttention(config=cfg, phantom_cfg=pcfg)
    assert attn.q_proj.weight.requires_grad is False
    assert attn.q_proj.lora_A.requires_grad is True
    assert attn.q_proj.lora_B.requires_grad is True

def test_phantom_distilbert_weight_copy():
    cfg = MockConfig()
    pcfg = PHANTOMConfig(cur_rank=8, lsh_rounds=2, lsh_bucket_size=2)
    old_attn = DummyDistilBertAttention(64)
    attn = PHANTOMAttention(config=cfg, phantom_cfg=pcfg, pretrained_attn=old_attn)
    x = torch.randn(2, 8, 64)
    out, _ = attn(x)
    assert out.shape == (2, 8, 64)

def test_phantom_roberta_weight_copy():
    cfg = MockConfig()
    pcfg = PHANTOMConfig(cur_rank=8, lsh_rounds=2, lsh_bucket_size=2)
    old_attn = DummyRobertaAttention(64)
    attn = PHANTOMAttention(config=cfg, phantom_cfg=pcfg, pretrained_attn=old_attn)
    assert attn.is_nested_encoder
    x = torch.randn(2, 8, 64)
    out, _ = attn(x)
    assert out.shape == (2, 8, 64)

def test_phantom_gpt2_weight_copy():
    cfg = MockConfig(is_decoder=True, model_type="gpt2")
    pcfg = PHANTOMConfig(cur_rank=8, lsh_rounds=2, lsh_bucket_size=2)
    old_attn = DummyGPT2Attention(64)
    attn = PHANTOMAttention(config=cfg, phantom_cfg=pcfg, pretrained_attn=old_attn)
    assert attn.is_decoder
    x = torch.randn(2, 8, 64)
    out, _ = attn(x)
    assert out.shape == (2, 8, 64)

def test_phantom_qwen_gqa_and_rope():
    cfg = MockConfig(hidden_size=64, num_attention_heads=4, num_key_value_heads=2, head_dim=16, is_decoder=True, model_type="qwen2")
    pcfg = PHANTOMConfig(cur_rank=8, lsh_rounds=2, lsh_bucket_size=2)
    old_attn = DummyQwenAttention(dim=64, num_kv_heads=2, head_dim=16)
    attn = PHANTOMAttention(config=cfg, phantom_cfg=pcfg, pretrained_attn=old_attn)
    assert attn.kv_repeat == 2
    x = torch.randn(2, 8, 64)
    cos = torch.randn(2, 8, 16)
    sin = torch.randn(2, 8, 16)
    out, _ = attn(x, position_embeddings=(cos, sin))
    assert out.shape == (2, 8, 64)

def test_phantom_inject_safe():
    class DummyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = DummyDistilBertAttention(64)

    layer = DummyLayer()
    res = inject_phantom_attention(layer=layer, model_config=MockConfig(), old_attn=layer.attention)
    assert res is not None
    x = torch.randn(2, 8, 64)
    out, _ = layer.attention(x)
    assert out.shape == (2, 8, 64)


# ===========================================================================
# Targeted Tests for PHANTOM Bug Fixes
# ===========================================================================

def test_phantom_decoder_kv_cache_and_tuple_layout():
    """Verify autoregressive KV-caching and decoder return format."""
    cfg = MockConfig(is_decoder=True, model_type="gpt2")
    pcfg = PHANTOMConfig(cur_rank=8, lsh_rounds=1, lsh_bucket_size=2)
    attn = PHANTOMAttention(config=cfg, phantom_cfg=pcfg, hidden_size=64, num_heads=4)

    # Step 1: Prompt phase (L=4)
    x = torch.randn(2, 4, 64)
    res = attn(x, output_attentions=False)
    assert len(res) == 2, f"Expected (output, present_key_value), got len={len(res)}"
    out, past_kv = res
    assert out.shape == (2, 4, 64)
    assert isinstance(past_kv, tuple) and len(past_kv) == 2
    past_k, past_v = past_kv
    assert past_k.shape == (2, 4, 4, 16)
    assert past_v.shape == (2, 4, 4, 16)

    # Step 2: Next token generation (L=1) with past_key_value
    x_next = torch.randn(2, 1, 64)
    out_next, next_kv = attn(x_next, past_key_value=past_kv, output_attentions=False)
    assert out_next.shape == (2, 1, 64)
    assert next_kv[0].shape == (2, 4, 5, 16)
    assert next_kv[1].shape == (2, 4, 5, 16)

    # Step 3: Decoder with output_attentions=True returns (output, present_key_value, None)
    res_attn = attn(x_next, past_key_value=past_kv, output_attentions=True)
    assert len(res_attn) == 3
    assert res_attn[2] is None


def test_phantom_encoder_return_tuple():
    """Verify encoder returns (output, None)."""
    cfg = MockConfig(is_decoder=False, model_type="bert")
    pcfg = PHANTOMConfig(cur_rank=8, lsh_rounds=1)
    attn = PHANTOMAttention(config=cfg, phantom_cfg=pcfg, hidden_size=64, num_heads=4)
    x = torch.randn(2, 6, 64)
    res = attn(x)
    assert len(res) == 2
    out, cache = res
    assert out.shape == (2, 6, 64)
    assert cache is None


def test_phantom_robust_mask_handling():
    """Verify 2D, 3D, and float additive masks (-1e4) are handled cleanly."""
    cfg = MockConfig(is_decoder=True, model_type="gpt2")
    pcfg = PHANTOMConfig(cur_rank=8, lsh_rounds=2, lsh_bucket_size=2)
    attn = PHANTOMAttention(config=cfg, phantom_cfg=pcfg, hidden_size=64, num_heads=4)
    x = torch.randn(2, 6, 64)

    # 1. 2D binary float mask
    mask_2d = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 0.0, 0.0]])
    out1, _ = attn(x, attention_mask=mask_2d)
    assert out1.shape == (2, 6, 64)
    assert torch.isfinite(out1).all()

    # 2. 3D float additive mask (-1e4 for pad)
    mask_3d = torch.zeros(2, 1, 6)
    mask_3d[1, 0, 4:] = -1e4
    out2, _ = attn(x, attention_mask=mask_3d)
    assert out2.shape == (2, 6, 64)
    assert torch.isfinite(out2).all()


def test_phantom_cur_fallback_device_dtype():
    """Verify CUR fallback initializes on exact device and dtype."""
    from neuralops.attentions.phantom import FactorisedLinear_CUR
    linear = FactorisedLinear_CUR(in_features=64, out_features=64, cur_rank=8)
    
    # Pass a tensor with NaNs to trigger except block
    bad_weight = torch.full((64, 64), float('nan'))
    bias = torch.randn(64)
    linear.copy_cur(bad_weight, bias=bias)

    assert linear.down.weight.device == bad_weight.device
    assert torch.isfinite(linear.down.weight).all()
    assert torch.isfinite(linear.up.weight).all()
    assert torch.isfinite(linear.up.bias).all()

