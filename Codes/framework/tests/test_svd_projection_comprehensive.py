import types

import pytest
import torch
import torch.nn as nn
from framework.attentions.svd_proj_attn import \
    SVDProjectionAttention as SVDProjectionAttentionAlias
from framework.attentions.svd_proj_attn import \
    inject_svd_proj_attn as inject_svd_proj_attn_alias
from framework.attentions.svd_projection_attention import (
    SVDProjectionAttention, SVDProjectionLinear, inject_svd_proj_attn)


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


def test_svd_projection_properties_and_alias():
    assert SVDProjectionAttention is SVDProjectionAttentionAlias
    assert inject_svd_proj_attn is inject_svd_proj_attn_alias
    cfg = make_config(hidden_size=64, num_heads=4, num_kv_heads=2)
    attn = SVDProjectionAttention(cfg, old_attn=DummyAttn(64, 4))
    assert attn.hidden_size == 64
    assert attn.num_heads == 4
    assert attn.num_kv_heads == 2
    assert hasattr(attn, "q_proj")
    assert hasattr(attn, "k_proj")
    assert hasattr(attn, "v_proj")
    assert hasattr(attn, "out_proj")


def test_svd_projection_forward_and_shapes():
    cfg = make_config(hidden_size=64, num_heads=4, num_kv_heads=2)
    attn = SVDProjectionAttention(cfg, old_attn=DummyAttn(64, 4))
    attn.eval()

    for seq_len in [2, 8, 16, 25]:
        B, L, D = 2, seq_len, 64
        x = torch.randn(B, L, D)
        out, _ = attn(x)
        assert out.shape == (B, L, D), f"Expected shape {(B, L, D)}, got {out.shape}"
        assert not torch.isnan(out).any()


def test_svd_projection_causal_mode():
    cfg = make_config(hidden_size=64, num_heads=4, num_kv_heads=2, is_decoder=True, model_type="gpt2")
    attn = SVDProjectionAttention(cfg, is_decoder=True, old_attn=DummyAttn(64, 4))
    attn.eval()

    B, L, D = 2, 16, 64
    x = torch.randn(B, L, D)
    out, _ = attn(x)
    assert out.shape == (B, L, D)
    assert not torch.isnan(out).any()


def test_svd_projection_copy_weights_all_4_architectures():
    d = 64
    h = 4
    cfg = make_config(hidden_size=d, num_heads=h, num_kv_heads=h)

    # 1. DistilBERT style: q_lin, k_lin, v_lin, out_lin
    class DistilBertDummy(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_lin = nn.Linear(d, d)
            self.k_lin = nn.Linear(d, d)
            self.v_lin = nn.Linear(d, d)
            self.out_lin = nn.Linear(d, d)

    distil_attn = DistilBertDummy()
    attn1 = SVDProjectionAttention(cfg, old_attn=distil_attn)
    assert attn1 is not None

    # 2. RoBERTa style: self.query, self.key, self.value, output.dense
    class RobertaDummy(nn.Module):
        def __init__(self):
            super().__init__()
            self.self = nn.Module()
            self.self.query = nn.Linear(d, d)
            self.self.key = nn.Linear(d, d)
            self.self.value = nn.Linear(d, d)
            self.output = nn.Module()
            self.output.dense = nn.Linear(d, d)

    roberta_attn = RobertaDummy()
    attn2 = SVDProjectionAttention(cfg, old_attn=roberta_attn)
    assert attn2 is not None

    # 3. GPT-2 style: c_attn, c_proj (Conv1D-like)
    class Conv1D(nn.Module):
        def __init__(self, nf, nx):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(nx, nf))
            self.bias = nn.Parameter(torch.zeros(nf))

    class GPT2Dummy(nn.Module):
        def __init__(self):
            super().__init__()
            self.c_attn = Conv1D(3 * d, d)
            self.c_proj = Conv1D(d, d)

    gpt2_attn = GPT2Dummy()
    attn3 = SVDProjectionAttention(cfg, old_attn=gpt2_attn)
    assert attn3 is not None

    # 4. Qwen / LLaMA style: q_proj, k_proj, v_proj, o_proj
    class QwenDummy(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(d, d)
            self.k_proj = nn.Linear(d, d)
            self.v_proj = nn.Linear(d, d)
            self.o_proj = nn.Linear(d, d)

    qwen_attn = QwenDummy()
    attn4 = SVDProjectionAttention(cfg, old_attn=qwen_attn)
    assert attn4 is not None


def test_svd_projection_injection():
    d = 64
    h = 4
    cfg = make_config(hidden_size=d, num_heads=h, num_kv_heads=h)

    class LayerDummy(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = nn.Module()
            self.attention.q_lin = nn.Linear(d, d)
            self.attention.k_lin = nn.Linear(d, d)
            self.attention.v_lin = nn.Linear(d, d)
            self.attention.out_lin = nn.Linear(d, d)

    layer = LayerDummy()
    res = inject_svd_proj_attn(layer=layer, model_config=cfg)
    assert res is not None
    assert isinstance(layer.attention, SVDProjectionAttention)

    # Test whole-model injection
    class DummyTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = cfg
            self.transformer = nn.Module()
            self.transformer.h = nn.ModuleList([LayerDummy() for _ in range(2)])

    model = DummyTransformer()
    res_m = inject_svd_proj_attn(layer=model, model_config=cfg)
    assert res_m is not None
    for lyr in model.transformer.h:
        assert isinstance(lyr.attention, SVDProjectionAttention)
