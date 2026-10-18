from .testing_utils import MockConfig, DummyDistilBertAttention
import pytest
import torch
import torch.nn as nn
from neuralops.attentions.hymba_transx import (HymbaTransXBlock,
                                               inject_hymba_transx)

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

def test_hymba_transx_basic():
    cfg = MockConfig()
    attn = HymbaTransXBlock(config=cfg, hidden_size=64, num_heads=4)
    x = torch.randn(2, 8, 64)
    out, _ = attn(x)
    assert out.shape == (2, 8, 64)

def test_hymba_transx_lora_trainable():
    cfg = MockConfig(lora_rank=8)
    attn = HymbaTransXBlock(config=cfg, hidden_size=64, num_heads=4)
    assert attn.W_q1.weight.requires_grad is False
    assert attn.W_q1.lora_A.requires_grad is True
    assert attn.W_q1.lora_B.requires_grad is True

def test_hymba_transx_distilbert_weight_copy():
    cfg = MockConfig()
    old_attn = DummyDistilBertAttention(64)
    attn = HymbaTransXBlock(config=cfg, hidden_size=64, num_heads=4, old_attn=old_attn)
    x = torch.randn(2, 8, 64)
    out, _ = attn(x)
    assert out.shape == (2, 8, 64)

def test_hymba_transx_roberta_weight_copy():
    cfg = MockConfig()
    old_attn = DummyRobertaAttention(64)
    attn = HymbaTransXBlock(config=cfg, hidden_size=64, num_heads=4, old_attn=old_attn)
    assert attn.is_nested_encoder
    x = torch.randn(2, 8, 64)
    out, _ = attn(x)
    assert out.shape == (2, 8, 64)

def test_hymba_transx_gpt2_weight_copy():
    cfg = MockConfig(is_decoder=True, model_type="gpt2")
    old_attn = DummyGPT2Attention(64)
    attn = HymbaTransXBlock(config=cfg, hidden_size=64, num_heads=4, old_attn=old_attn)
    assert attn.is_decoder
    x = torch.randn(2, 8, 64)
    out, _ = attn(x)
    assert out.shape == (2, 8, 64)

def test_hymba_transx_qwen_gqa_and_rope():
    cfg = MockConfig(hidden_size=64, num_attention_heads=4, num_key_value_heads=2, head_dim=16, is_decoder=True, model_type="qwen2")
    old_attn = DummyQwenAttention(dim=64, num_kv_heads=2, head_dim=16)
    attn = HymbaTransXBlock(config=cfg, hidden_size=64, num_heads=4, old_attn=old_attn)
    assert attn.kv_repeat == 2
    x = torch.randn(2, 8, 64)
    cos = torch.randn(2, 8, 16)
    sin = torch.randn(2, 8, 16)
    out, _ = attn(x, position_embeddings=(cos, sin))
    assert out.shape == (2, 8, 64)

def test_hymba_transx_inject_safe():
    class DummyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = DummyDistilBertAttention(64)

    layer = DummyLayer()
    res = inject_hymba_transx(layer=layer, model_config=MockConfig())
    assert res is not None
    x = torch.randn(2, 8, 64)
    out, _ = layer.attention(x)
    assert out.shape == (2, 8, 64)


# ── FIX-1: KV-cache return tuple ───────────────────────────────────────────
def test_hymba_transx_decoder_kv_cache_return():
    """Decoder must return (out, present_key_value) where present is (k1, v, k2)."""
    cfg = MockConfig(is_decoder=True, model_type="gpt2")
    attn = HymbaTransXBlock(config=cfg, hidden_size=64, num_heads=4)
    x = torch.randn(2, 4, 64)
    result = attn(x)
    # Decoder without output_attentions → 2-tuple
    assert len(result) == 2, f"Expected 2-tuple, got {len(result)}"
    out, present = result
    assert out.shape == (2, 4, 64)
    # present_key_value must be a 3-tuple of tensors
    assert isinstance(present, tuple) and len(present) == 3, (
        f"present_key_value should be (k1, v, k2), got {type(present)}"
    )
    k1, v, k2 = present
    assert k1.shape[-2] == 4 and v.shape[-2] == 4 and k2.shape[-2] == 4


def test_hymba_transx_decoder_kv_cache_output_attentions():
    """Decoder with output_attentions=True must return 3-tuple (out, present_kv, attn_weights)."""
    cfg = MockConfig(is_decoder=True, model_type="gpt2")
    attn = HymbaTransXBlock(config=cfg, hidden_size=64, num_heads=4)
    x = torch.randn(2, 4, 64)
    result = attn(x, output_attentions=True)
    assert len(result) == 3, f"Expected 3-tuple, got {len(result)}"
    out, present, attn_weights = result
    assert out.shape == (2, 4, 64)
    assert attn_weights is not None


def test_hymba_transx_encoder_no_present_kv():
    """Encoder (non-decoder) must return (out, None) regardless of output_attentions=False."""
    cfg = MockConfig(is_decoder=False, model_type="bert")
    attn = HymbaTransXBlock(config=cfg, hidden_size=64, num_heads=4)
    x = torch.randn(2, 8, 64)
    out, present = attn(x)
    assert out.shape == (2, 8, 64)
    assert present is None, "Encoder should not return a KV cache"


# ── FIX-2: 3D attention mask ────────────────────────────────────────────────
def test_hymba_transx_3d_bool_mask():
    """3D boolean attention mask (B, 1, L) must not raise ValueError."""
    cfg = MockConfig(is_decoder=True, model_type="gpt2")
    attn = HymbaTransXBlock(config=cfg, hidden_size=64, num_heads=4)
    x = torch.randn(2, 6, 64)
    mask3d = torch.ones(2, 1, 6, dtype=torch.bool)
    out, _ = attn(x, attention_mask=mask3d)
    assert out.shape == (2, 6, 64)


def test_hymba_transx_3d_float_mask():
    """3D float attention mask (B, 1, L) with additive-style values must work."""
    cfg = MockConfig(is_decoder=False, model_type="bert")
    attn = HymbaTransXBlock(config=cfg, hidden_size=64, num_heads=4)
    x = torch.randn(2, 6, 64)
    mask3d = torch.zeros(2, 1, 6)  # 0 = valid
    out, _ = attn(x, attention_mask=mask3d)
    assert out.shape == (2, 6, 64)


# ── FIX-3: safe diff_attn when relu output is all-zero ─────────────────────
def test_hymba_transx_safe_diff_attn_no_nan():
    """When all ReLU(attn1 - lam*attn2) == 0, output must be finite (no NaN/Inf)."""
    cfg = MockConfig(is_decoder=True, model_type="gpt2")
    attn = HymbaTransXBlock(config=cfg, hidden_size=64, num_heads=4)
    # Force diff_lambda very large so lam→1 and attn2 ≈ attn1 → ReLU → 0
    with torch.no_grad():
        attn.diff_lambda.fill_(50.0)
    x = torch.randn(2, 4, 64)
    out, _ = attn(x)
    assert torch.isfinite(out).all(), "Output contains NaN/Inf after safe diff_attn fix"

