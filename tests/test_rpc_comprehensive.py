import types

import pytest
import torch
import torch.nn as nn
from neuralops.attentions.rpc_attention import (LoRALinear, RPCAttention,
                                                _rotate_half,
                                                apply_rotary_pos_emb,
                                                inject_rpc)


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


def test_rpc_properties_and_clamping():
    cfg = make_config(hidden_size=64, num_heads=4, num_kv_heads=2)
    # Head dim is 16. Pass rank 64, should be clamped to 16
    attn = RPCAttention(cfg, rank=64, pretrained_attn=DummyAttn(64, 4))
    assert attn.rank == 16, f"Expected rank clamped to head_dim=16, got {attn.rank}"
    assert attn.proj.shape == (4, 16, 16)


def test_rpc_forward_and_shapes():
    cfg = make_config(hidden_size=64, num_heads=4, num_kv_heads=2)
    attn = RPCAttention(cfg, rank=8, pretrained_attn=DummyAttn(64, 4))
    attn.eval()

    for seq_len in [2, 8, 16, 25]:
        B, L, D = 2, seq_len, 64
        x = torch.randn(B, L, D)
        out, _ = attn(x)
        assert out.shape == (B, L, D), f"Expected shape {(B, L, D)}, got {out.shape}"
        assert not torch.isnan(out).any()


def test_rpc_causal_mode():
    cfg = make_config(hidden_size=64, num_heads=4, num_kv_heads=2, is_decoder=True, model_type="gpt2")
    attn = RPCAttention(cfg, rank=8, is_decoder=True, pretrained_attn=DummyAttn(64, 4))
    attn.eval()

    B, L, D = 2, 16, 64
    x = torch.randn(B, L, D)
    out, _ = attn(x)
    assert out.shape == (B, L, D)
    assert not torch.isnan(out).any()


def test_rpc_rotate_half_and_odd_dim():
    for d in [16, 17, 31, 32]:
        t = torch.randn(2, 4, 8, d)
        rotated = _rotate_half(t)
        assert rotated.shape == t.shape, f"Odd dim {d} failed in _rotate_half"

    head_dim_odd = 15
    q = torch.randn(2, 4, 8, head_dim_odd)
    k = torch.randn(2, 2, 8, head_dim_odd)
    cos = torch.randn(8, head_dim_odd)
    sin = torch.randn(8, head_dim_odd)

    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape
    assert not torch.isnan(q_rot).any()


def test_rpc_lora_and_gradients():
    cfg = make_config(hidden_size=64, num_heads=4, num_kv_heads=2)
    attn = RPCAttention(cfg, rank=8, pretrained_attn=DummyAttn(64, 4), lora_rank=8)
    attn.train()

    # Base frozen, lora trainable
    assert not attn.W_q.weight.requires_grad
    assert attn.W_q.lora_A.requires_grad
    assert attn.W_q.lora_B.requires_grad

    x = torch.randn(2, 8, 64, requires_grad=True)
    out, _ = attn(x)
    loss = out.sum()
    loss.backward()

    assert attn.W_q.lora_A.grad is not None
    assert attn.W_q.lora_B.grad is not None
    assert attn.W_q.weight.grad is None


def test_rpc_inject():
    class DummyLayer(nn.Module):
        def __init__(self, d=64, h=4):
            super().__init__()
            self.attention = DummyAttn(d, h)

    layer = DummyLayer(64, 4)
    cfg = make_config(hidden_size=64, num_heads=4, num_kv_heads=2)
    injected = inject_rpc(layer, model_config=cfg, old_attn=layer.attention)
    assert isinstance(injected, RPCAttention)
    assert layer.attention is injected

    empty_layer = nn.Module()
    assert inject_rpc(empty_layer, model_config=cfg, old_attn=None) is None
