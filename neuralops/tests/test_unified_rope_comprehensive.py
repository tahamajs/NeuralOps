import types

import pytest
import torch
import torch.nn as nn
from neuralops.attentions.unified_rope import (LoRALinear, UnifiedRoPE,
                                               _rotate_half,
                                               apply_rotary_pos_emb,
                                               inject_unified_rope)


def make_config(hidden_size=64, num_heads=4, num_kv_heads=2, is_decoder=False, model_type="bert"):
    return types.SimpleNamespace(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=hidden_size // num_heads,
        is_decoder=is_decoder,
        model_type=model_type,
        dropout=0.0,
        lora_rank=16,
    )


class DummyAttn(nn.Module):
    def __init__(self, d=64, h=4):
        super().__init__()
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.o_proj = nn.Linear(d, d)
        self.hidden_size = d
        self.num_heads = h


def test_unified_rope_forward_and_shapes():
    cfg = make_config(hidden_size=64, num_heads=4, num_kv_heads=2)
    attn = UnifiedRoPE(cfg, pretrained_attn=DummyAttn(64, 4))
    attn.eval()

    for seq_len in [4, 16, 32]:
        B, L, D = 2, seq_len, 64
        x = torch.randn(B, L, D)
        out, _ = attn(x)
        assert out.shape == (B, L, D), f"Expected shape {(B, L, D)}, got {out.shape}"
        assert not torch.isnan(out).any()


def test_unified_rope_causal_mode():
    cfg = make_config(hidden_size=64, num_heads=4, num_kv_heads=2, is_decoder=True, model_type="gpt2")
    attn = UnifiedRoPE(cfg, pretrained_attn=DummyAttn(64, 4), is_decoder=True)
    attn.eval()

    B, L, D = 2, 16, 64
    x = torch.randn(B, L, D)
    out, _ = attn(x)
    assert out.shape == (B, L, D)
    assert not torch.isnan(out).any()


def test_unified_rope_rotate_half_and_odd_dim():
    for d in [16, 17, 31, 32]:
        t = torch.randn(2, 4, 8, d)
        rotated = _rotate_half(t)
        assert rotated.shape == t.shape, f"Odd dim {d} failed shape preservation in _rotate_half"

    head_dim_odd = 15
    q = torch.randn(2, 4, 8, head_dim_odd)
    k = torch.randn(2, 2, 8, head_dim_odd)
    cos = torch.randn(8, head_dim_odd)
    sin = torch.randn(8, head_dim_odd)

    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape
    assert not torch.isnan(q_rot).any()


def test_unified_rope_masks():
    cfg = make_config(hidden_size=64, num_heads=4, num_kv_heads=2)
    attn = UnifiedRoPE(cfg, pretrained_attn=DummyAttn(64, 4))
    attn.eval()

    B, L, D = 2, 12, 64
    x = torch.randn(B, L, D)

    # 2D mask
    mask_2d = torch.ones(B, L)
    mask_2d[:, -3:] = 0
    out_2d, _ = attn(x, attention_mask=mask_2d)
    assert out_2d.shape == (B, L, D)
    assert not torch.isnan(out_2d).any()

    # 4D mask
    mask_4d = torch.zeros(B, 1, L, L)
    mask_4d[:, :, :, -3:] = -10000.0
    out_4d, _ = attn(x, attention_mask=mask_4d)
    assert out_4d.shape == (B, L, D)
    assert not torch.isnan(out_4d).any()


def test_unified_rope_lora_and_gradients():
    cfg = make_config(hidden_size=64, num_heads=4, num_kv_heads=2)
    attn = UnifiedRoPE(cfg, pretrained_attn=DummyAttn(64, 4), lora_rank=8)
    attn.train()

    # Base frozen, lora trainable
    assert not attn.q_proj.weight.requires_grad
    assert attn.q_proj.lora_A.requires_grad
    assert attn.q_proj.lora_B.requires_grad

    x = torch.randn(2, 8, 64, requires_grad=True)
    out, _ = attn(x)
    loss = out.sum()
    loss.backward()

    assert attn.q_proj.lora_A.grad is not None
    assert attn.q_proj.lora_B.grad is not None
    assert attn.q_proj.weight.grad is None


def test_unified_rope_inject():
    class DummyLayer(nn.Module):
        def __init__(self, d=64, h=4):
            super().__init__()
            self.attention = DummyAttn(d, h)

    layer = DummyLayer(64, 4)
    cfg = make_config(hidden_size=64, num_heads=4, num_kv_heads=2)
    injected = inject_unified_rope(layer, model_config=cfg, old_attn=layer.attention)
    assert isinstance(injected, UnifiedRoPE)
    assert layer.attention is injected

    # Test None old_attn graceful handling
    res = inject_unified_rope(layer, model_config=cfg, old_attn=None)
    assert isinstance(res, UnifiedRoPE)  # finds layer.attention
    empty_layer = nn.Module()
    assert inject_unified_rope(empty_layer, model_config=cfg, old_attn=None) is None
