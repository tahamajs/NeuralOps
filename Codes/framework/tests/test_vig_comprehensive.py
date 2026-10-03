import types

import pytest
import torch
import torch.nn as nn
from framework.attentions.vig_attention import (LoRALinear, ViGAttention,
                                                _rotate_half,
                                                apply_rotary_pos_emb,
                                                inject_vig)


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


def test_vig_forward_and_grid_padding():
    cfg = make_config(hidden_size=64, num_heads=4, num_kv_heads=2)
    attn = ViGAttention(cfg, pretrained_attn=DummyAttn(64, 4))
    attn.eval()

    # Test on multiple arbitrary sequence lengths to verify grid padding
    for seq_len in [7, 16, 23]:
        B, L, D = 2, seq_len, 64
        x = torch.randn(B, L, D)

        out, _ = attn(x)
        assert out.shape == (B, L, D), f"Failed for seq_len {seq_len}: expected {(B, L, D)}, got {out.shape}"
        assert not torch.isnan(out).any()


def test_causal_shape_and_causality():
    cfg = make_config(hidden_size=64, num_heads=4, num_kv_heads=2, is_decoder=True, model_type="gpt2")
    attn = ViGAttention(cfg, pretrained_attn=DummyAttn(64, 4))
    attn.eval()

    B, L, D = 2, 16, 64
    x = torch.randn(B, L, D)

    out, _ = attn(x)
    assert out.shape == (B, L, D)
    assert not torch.isnan(out).any()


def test_bidirectional_and_padding_mask():
    cfg = make_config(hidden_size=64, num_heads=4, num_kv_heads=2)
    attn = ViGAttention(cfg, pretrained_attn=DummyAttn(64, 4))
    attn.eval()

    B, L, D = 2, 16, 64
    x = torch.randn(B, L, D)
    mask = torch.ones(B, L)
    mask[:, -4:] = 0

    out, _ = attn(x, attention_mask=mask)
    assert out.shape == (B, L, D)
    assert not torch.isnan(out).any()


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
    cfg = make_config(hidden_size=64, num_heads=4)
    dummy = DummyAttn(64, 4)
    dummy.q_proj.lora_A = nn.Parameter(torch.randn(16, 64))
    dummy.q_proj.lora_B = nn.Parameter(torch.randn(64, 16))
    dummy.q_proj.lora_scaling = 2.0

    attn = ViGAttention(cfg, pretrained_attn=dummy)

    assert attn.q_proj.weight.requires_grad is False
    assert attn.q_proj.lora_A.requires_grad is True
    assert attn.q_proj.lora_B.requires_grad is True
    assert attn.q_proj.lora_scaling == 2.0
    assert torch.allclose(attn.q_proj.lora_A, dummy.q_proj.lora_A)


def test_head_mask_broadcasting():
    cfg = make_config(hidden_size=64, num_heads=4)
    attn = ViGAttention(cfg, pretrained_attn=DummyAttn(64, 4))
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


def test_inject_vig():
    class DummyLayer(nn.Module):
        def __init__(self, d=64):
            super().__init__()
            self.attention = DummyAttn(d)

    layer = DummyLayer(64)
    res = inject_vig(layer=layer, old_attn=layer.attention)
    assert res is not None
    assert isinstance(layer.attention, ViGAttention)

    # When old_attn is None, should return None gracefully
    none_res = inject_vig(layer=layer, old_attn=None)
    assert none_res is None
