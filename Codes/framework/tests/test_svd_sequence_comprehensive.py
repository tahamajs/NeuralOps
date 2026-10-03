import types

import pytest
import torch
import torch.nn as nn
from framework.attentions.svd_attention import (LowRankSVDSequenceAttention,
                                                SVDLinearAttention,
                                                _rotate_half,
                                                apply_rotary_pos_emb,
                                                inject_svd_linear_attention)
from framework.attentions.svd_sequence_attention import \
    LowRankSVDSequenceAttention as LowRankSVDSequenceAttentionAlias
from framework.attentions.svd_sequence_attention import \
    SVDLinearAttention as SVDLinearAttentionAlias


def make_config(hidden_size=64, num_heads=4, num_kv_heads=2, is_decoder=False, model_type="bert"):
    return types.SimpleNamespace(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=hidden_size // num_heads,
        is_decoder=is_decoder,
        model_type=model_type,
        dropout=0.0,
        lora_rank=8,
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


def test_svd_sequence_properties_and_alias():
    assert SVDLinearAttention is SVDLinearAttentionAlias
    assert LowRankSVDSequenceAttention is LowRankSVDSequenceAttentionAlias
    cfg = make_config(hidden_size=64, num_heads=4, num_kv_heads=2)
    attn = SVDLinearAttention(cfg, pretrained_attn=DummyAttn(64, 4), rank=8)
    assert attn.rank == 8


def test_svd_sequence_forward_and_shapes():
    cfg = make_config(hidden_size=64, num_heads=4, num_kv_heads=2)
    attn = SVDLinearAttention(cfg, pretrained_attn=DummyAttn(64, 4), rank=8)
    attn.eval()

    for seq_len in [2, 8, 16, 25]:
        B, L, D = 2, seq_len, 64
        x = torch.randn(B, L, D)
        out, _ = attn(x)
        assert out.shape == (B, L, D), f"Expected shape {(B, L, D)}, got {out.shape}"
        assert not torch.isnan(out).any()


def test_svd_sequence_causal_mode():
    cfg = make_config(hidden_size=64, num_heads=4, num_kv_heads=2, is_decoder=True, model_type="gpt2")
    attn = SVDLinearAttention(cfg, is_decoder=True, pretrained_attn=DummyAttn(64, 4), rank=8)
    attn.eval()

    B, L, D = 2, 16, 64
    x = torch.randn(B, L, D)
    out, _ = attn(x)
    assert out.shape == (B, L, D)
    assert not torch.isnan(out).any()


def test_svd_sequence_rotate_half_and_odd_dim():
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


def test_svd_sequence_lora_and_gradients():
    cfg = make_config(hidden_size=64, num_heads=4, num_kv_heads=2)
    attn = SVDLinearAttention(cfg, pretrained_attn=DummyAttn(64, 4), lora_rank=8, rank=8)
    attn.train()

    # Base frozen, lora trainable
    assert not attn.query.weight.requires_grad
    assert attn.query.lora_A.requires_grad
    assert attn.query.lora_B.requires_grad

    x = torch.randn(2, 8, 64, requires_grad=True)
    out, _ = attn(x)
    loss = out.sum()
    loss.backward()

    assert attn.query.lora_A.grad is not None
    assert attn.query.lora_B.grad is not None
    assert attn.query.weight.grad is None


def test_svd_sequence_injector_and_architectures():
    # 1. DistilBERT pattern
    class DistilAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_lin = nn.Linear(64, 64)
            self.k_lin = nn.Linear(64, 64)
            self.v_lin = nn.Linear(64, 64)
            self.out_lin = nn.Linear(64, 64)

    class DistilLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = DistilAttn()

    l_distil = DistilLayer()
    cfg_distil = types.SimpleNamespace(hidden_size=64, num_attention_heads=4, head_dim=16)
    res = inject_svd_linear_attention(l_distil, config=cfg_distil, comp_config={"rank": 8})
    assert res is not None
    assert isinstance(l_distil.attention, SVDLinearAttention)

    # 2. RoBERTa pattern
    class RobertaSelf(nn.Module):
        def __init__(self):
            super().__init__()
            self.query = nn.Linear(64, 64)
            self.key = nn.Linear(64, 64)
            self.value = nn.Linear(64, 64)

    class RobertaOut(nn.Module):
        def __init__(self):
            super().__init__()
            self.dense = nn.Linear(64, 64)

    class RobertaLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = types.SimpleNamespace(self=RobertaSelf(), output=RobertaOut())

    l_roberta = RobertaLayer()
    cfg_roberta = types.SimpleNamespace(hidden_size=64, num_attention_heads=4, head_dim=16)
    res = inject_svd_linear_attention(l_roberta.attention, config=cfg_roberta, comp_config={"rank": 8})
    assert res is not None

    # 3. GPT-2 pattern
    class GPT2Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.c_attn = nn.Linear(64, 192)
            self.c_proj = nn.Linear(64, 64)

    class GPT2Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = GPT2Attn()

    l_gpt2 = GPT2Block()
    cfg_gpt2 = types.SimpleNamespace(hidden_size=64, num_attention_heads=4, head_dim=16, is_decoder=True, model_type="gpt2")
    res = inject_svd_linear_attention(l_gpt2, config=cfg_gpt2, comp_config={"rank": 8})
    assert res is not None
    assert isinstance(l_gpt2.attn, SVDLinearAttention)

    # 4. Qwen pattern
    class QwenAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(64, 64)
            self.k_proj = nn.Linear(64, 32)
            self.v_proj = nn.Linear(64, 32)
            self.o_proj = nn.Linear(64, 64)

    class QwenDecoderLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = QwenAttn()

    l_qwen = QwenDecoderLayer()
    cfg_qwen = types.SimpleNamespace(hidden_size=64, num_attention_heads=4, num_key_value_heads=2, head_dim=16, is_decoder=True, model_type="qwen2")
    res = inject_svd_linear_attention(l_qwen, config=cfg_qwen, comp_config={"rank": 8})
    assert res is not None
    assert isinstance(l_qwen.self_attn, SVDLinearAttention)
