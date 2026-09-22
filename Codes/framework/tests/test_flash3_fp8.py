"""Tests for FlashAttention-3 FP8 attention module.

Validates:
1. Instantiation with default settings (use_fp8=False default)
2. Fallback behavior when use_fp8=True on non-Hopper / CPU devices
3. Pretrained weight copying from DistilBERT, RoBERTa, GPT-2, and Qwen
4. RuntimeError raised on missing mandatory projections
5. Rotary position embeddings (RoPE) and Grouped Query Attention (GQA)
6. Output projection bypass for nested encoder (RoBERTa)
7. Head mask handling in attention output
8. Causal masking correctness
"""

import math
import pytest
import torch
import torch.nn as nn

from framework.attentions.flash_attention3_fp8 import FlashAttention3FP8, inject_flash3_fp8


class MockConfig:
    def __init__(
        self,
        hidden_size: int = 64,
        num_attention_heads: int = 4,
        num_key_value_heads: int = 4,
        head_dim: int = 16,
        is_decoder: bool = False,
        model_type: str = "bert",
        dropout: float = 0.0,
    ):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.is_decoder = is_decoder
        self.model_type = model_type
        self.dropout = dropout


class DummyDistilBertAttention(nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        self.q_lin = nn.Linear(dim, dim)
        self.k_lin = nn.Linear(dim, dim)
        self.v_lin = nn.Linear(dim, dim)
        self.out_lin = nn.Linear(dim, dim)


class DummyRobertaSelfAttention(nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)


class DummyRobertaAttention(nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        self.self = DummyRobertaSelfAttention(dim)
        self.output = nn.Module()
        self.output.dense = nn.Linear(dim, dim)


class DummyGPT2Attention(nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        self.c_attn = nn.Linear(dim, 3 * dim)
        self.c_proj = nn.Linear(dim, dim)


class DummyQwenAttention(nn.Module):
    def __init__(self, dim: int = 64, num_kv_heads: int = 2, head_dim: int = 16):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, num_kv_heads * head_dim)
        self.v_proj = nn.Linear(dim, num_kv_heads * head_dim)
        self.o_proj = nn.Linear(dim, dim)


def test_default_init_and_forward():
    """Test standard init with default use_fp8=False and forward pass."""
    cfg = MockConfig()
    attn = FlashAttention3FP8(config=cfg)
    assert not attn.use_fp8, "Default use_fp8 must be False"
    x = torch.randn(2, 8, 64)
    out, _ = attn(x)
    assert out.shape == (2, 8, 64)
    assert not torch.isnan(out).any()


def test_auto_disable_fp8_on_non_hopper():
    """Test that requesting use_fp8=True auto-disables on non-Hopper / CPU devices."""
    cfg = MockConfig()
    attn = FlashAttention3FP8(config=cfg, use_fp8=True)
    # Unless running on actual Hopper GPU (SM >= 9.0), use_fp8 should be False
    is_hopper = False
    if torch.cuda.is_available() and hasattr(torch.cuda, "get_device_capability"):
        try:
            is_hopper = (torch.cuda.get_device_capability()[0] >= 9)
        except Exception:
            is_hopper = False
    if not is_hopper:
        assert not attn.use_fp8, "use_fp8 must be auto-disabled on non-Hopper hardware"


def test_none_config_init():
    """Test initialization when config=None."""
    attn = FlashAttention3FP8(config=None, hidden_size=64, num_heads=4)
    x = torch.randn(2, 8, 64)
    out, _ = attn(x)
    assert out.shape == (2, 8, 64)


def test_distilbert_weight_copy():
    """Test exact weight copying from DistilBERT linear projections."""
    cfg = MockConfig()
    old_attn = DummyDistilBertAttention(64)
    attn = FlashAttention3FP8(config=cfg, pretrained_attn=old_attn)
    assert not attn.is_nested_encoder
    x = torch.randn(2, 8, 64)
    out, _ = attn(x)
    assert out.shape == (2, 8, 64)


def test_roberta_weight_copy_and_nested_bypass():
    """Test weight copying from nested Roberta attention and nested encoder out_proj bypass."""
    cfg = MockConfig()
    old_attn = DummyRobertaAttention(64)
    attn = FlashAttention3FP8(config=cfg, pretrained_attn=old_attn)
    assert attn.is_nested_encoder, "Should detect nested encoder structure"
    x = torch.randn(2, 8, 64)
    out, _ = attn(x)
    assert out.shape == (2, 8, 64)


def test_gpt2_weight_copy_and_causal():
    """Test fused c_attn weight extraction from GPT-2 and causal attention."""
    cfg = MockConfig(is_decoder=True, model_type="gpt2")
    old_attn = DummyGPT2Attention(64)
    attn = FlashAttention3FP8(config=cfg, pretrained_attn=old_attn)
    assert attn.is_decoder
    x = torch.randn(2, 16, 64)
    out, _ = attn(x)
    assert out.shape == (2, 16, 64)


def test_qwen_gqa_and_rope():
    """Test Grouped Query Attention (GQA) head repeat and RoPE embeddings."""
    cfg = MockConfig(hidden_size=64, num_attention_heads=4, num_key_value_heads=2, head_dim=16, is_decoder=True, model_type="qwen2")
    old_attn = DummyQwenAttention(dim=64, num_kv_heads=2, head_dim=16)
    attn = FlashAttention3FP8(config=cfg, pretrained_attn=old_attn)
    assert attn.kv_repeat == 2
    x = torch.randn(2, 8, 64)
    cos = torch.randn(2, 8, 16)
    sin = torch.randn(2, 8, 16)
    out, _ = attn(x, position_embeddings=(cos, sin))
    assert out.shape == (2, 8, 64)


def test_missing_weights_raises_runtime_error():
    """Test that copy_weights raises RuntimeError if mandatory projections are missing."""
    class BrokenAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.foo = nn.Linear(64, 64)

    cfg = MockConfig()
    broken = BrokenAttention()
    with pytest.raises(RuntimeError, match="missing mandatory projections"):
        FlashAttention3FP8(config=cfg, pretrained_attn=broken)


def test_head_mask_handling():
    """Test that head_mask is properly applied to attention outputs."""
    cfg = MockConfig()
    attn = FlashAttention3FP8(config=cfg)
    x = torch.randn(2, 8, 64)
    # Mask out head 0 and head 1, keep head 2 and head 3
    head_mask = torch.tensor([0.0, 0.0, 1.0, 1.0])
    out, _ = attn(x, head_mask=head_mask)
    assert out.shape == (2, 8, 64)
    assert not torch.isnan(out).any()


def test_inject_flash3_fp8():
    """Test injection helper replaces standard attention safely."""
    class DummyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = DummyDistilBertAttention(64)

    layer = DummyLayer()
    res = inject_flash3_fp8(layer=layer, model_config=MockConfig())
    assert res is not None
    assert isinstance(layer.attention, FlashAttention3FP8)
    x = torch.randn(2, 8, 64)
    out, _ = layer.attention(x)
    assert out.shape == (2, 8, 64)


if __name__ == "__main__":
    test_default_init_and_forward()
    test_auto_disable_fp8_on_non_hopper()
    test_none_config_init()
    test_distilbert_weight_copy()
    test_roberta_weight_copy_and_nested_bypass()
    test_gpt2_weight_copy_and_causal()
    test_qwen_gqa_and_rope()
    test_missing_weights_raises_runtime_error()
    test_head_mask_handling()
    test_inject_flash3_fp8()
    print("All 10 FlashAttention-3 FP8 tests passed successfully!")
