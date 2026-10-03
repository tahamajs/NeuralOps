import types

import pytest
import torch
import torch.nn as nn
from framework.attentions.wgqa_attention import (LoRALinear,
                                                 WeightedGroupedQueryAttention,
                                                 _rotate_half,
                                                 apply_rotary_pos_emb,
                                                 inject_wgqa)


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


def test_wgqa_forward_and_head_weights():
    cfg = make_config(hidden_size=64, num_heads=4, num_kv_heads=2)
    dummy = DummyAttn(64, 4)
    attn = WeightedGroupedQueryAttention(cfg, pretrained_attn=dummy)
    attn.eval()

    B, L, D = 2, 8, 64
    x = torch.randn(B, L, D)

    # Check learnable importance weights exist
    assert hasattr(attn, "k_weights") and attn.k_weights.shape == (2, 1)
    assert hasattr(attn, "v_weights") and attn.v_weights.shape == (2, 1)

    out, attn_weights = attn(x, output_attentions=True)
    assert out.shape == (B, L, D)
    assert not torch.isnan(out).any()
    assert attn_weights.shape == (B, 4, L, L)


def test_causal_shape_and_causality():
    cfg = make_config(hidden_size=64, num_heads=4, num_kv_heads=2, is_decoder=True, model_type="gpt2")
    dummy = DummyAttn(64, 4)
    attn = WeightedGroupedQueryAttention(cfg, pretrained_attn=dummy)
    attn.eval()

    B, L, D = 2, 8, 64
    x = torch.randn(B, L, D)

    out, _ = attn(x)
    assert out.shape == (B, L, D)
    assert not torch.isnan(out).any()

    # Strict causality check: mutating tokens > l_probe should not alter output at <= l_probe
    l_probe = 3
    x_mod = x.clone()
    x_mod[:, l_probe + 1 :, :] = torch.randn_like(x_mod[:, l_probe + 1 :, :]) * 5.0 + 10.0

    out_orig, _ = attn(x)
    out_mod, _ = attn(x_mod)

    diff = torch.max(torch.abs(out_orig[:, : l_probe + 1, :] - out_mod[:, : l_probe + 1, :]))
    assert diff.item() < 1e-5, f"Causality violated! Max diff: {diff.item()}"


def test_gqa_pooling():
    # Source has 8 heads, WGQA configured for 2 KV heads
    cfg = make_config(hidden_size=64, num_heads=8, num_kv_heads=2)
    src = DummyAttn(64, 8)
    attn = WeightedGroupedQueryAttention(cfg, pretrained_attn=src)

    assert attn.num_heads == 8
    assert attn.num_kv_heads == 2
    assert attn.kv_repeat == 4
    # Check projection out_features
    assert attn.k_proj.out_features == 2 * (64 // 8)  # 16
    assert attn.v_proj.out_features == 2 * (64 // 8)  # 16


def test_rope_and_even_odd_dim_safety():
    for d in [16, 17, 63, 64]:
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


def test_lora_adapters_and_freezing():
    cfg = make_config(hidden_size=64, num_heads=4, num_kv_heads=2)
    dummy = DummyAttn(64, 4)
    dummy.q_proj.lora_A = nn.Parameter(torch.randn(16, 64))
    dummy.q_proj.lora_B = nn.Parameter(torch.randn(64, 16))
    dummy.q_proj.lora_scaling = 2.0

    attn = WeightedGroupedQueryAttention(cfg, pretrained_attn=dummy)

    assert attn.q_proj.weight.requires_grad is False
    assert attn.q_proj.lora_A.requires_grad is True
    assert attn.q_proj.lora_B.requires_grad is True
    assert attn.q_proj.lora_scaling == 2.0
    assert torch.allclose(attn.q_proj.lora_A, dummy.q_proj.lora_A)


def test_head_mask_broadcasting():
    cfg = make_config(hidden_size=64, num_heads=4, num_kv_heads=2)
    attn = WeightedGroupedQueryAttention(cfg, pretrained_attn=DummyAttn(64, 4))
    attn.eval()
    B, L, D, H = 2, 8, 64, 4
    x = torch.randn(B, L, D)

    # 1D: [H]
    hm1 = torch.ones(H)
    out1, _ = attn(x, head_mask=hm1)
    assert out1.shape == (B, L, D)

    # 2D: [B, H]
    hm2 = torch.ones(B, H)
    out2, _ = attn(x, head_mask=hm2)
    assert out2.shape == (B, L, D)

    # 3D: [B, H, L]
    hm3 = torch.ones(B, H, L)
    out3, _ = attn(x, head_mask=hm3)
    assert out3.shape == (B, L, D)

    # 4D: [B, H, L, L]
    hm4 = torch.ones(B, H, L, L)
    out4, _ = attn(x, head_mask=hm4)
    assert out4.shape == (B, L, D)


def test_inject_wgqa():
    class DummyLayer(nn.Module):
        def __init__(self, d=64):
            super().__init__()
            self.attention = DummyAttn(d)

    layer = DummyLayer(64)
    res = inject_wgqa(layer=layer, old_attn=layer.attention)
    assert res is not None
    assert isinstance(layer.attention, WeightedGroupedQueryAttention)

    # When old_attn is None, should return None gracefully
    none_res = inject_wgqa(layer=layer, old_attn=None)
    assert none_res is None
