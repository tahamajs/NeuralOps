import types

import pytest
import torch
import torch.nn as nn
from framework.attentions.spectra_v2 import (LoRALinear, SpectraConfig,
                                             SpectraV2Attention, _rotate_half,
                                             apply_rotary_pos_emb,
                                             inject_spectra_v2)
from framework.attentions.spectra_v2_attention import \
    SpectraV2Attention as SpectraV2AttentionAlias
from framework.attentions.spectra_v2_attention import \
    inject_spectra_v2 as inject_spectra_v2_alias


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


def test_spectra_v2_properties_and_alias():
    assert SpectraV2Attention is SpectraV2AttentionAlias
    assert inject_spectra_v2 is inject_spectra_v2_alias
    cfg = make_config(hidden_size=64, num_heads=4, num_kv_heads=2)
    attn = SpectraV2Attention(cfg, pretrained_attn=DummyAttn(64, 4))
    assert attn.hidden_size == 64
    assert attn.num_heads == 4
    assert attn.num_kv_heads == 2
    assert hasattr(attn, "query")
    assert hasattr(attn, "key")
    assert hasattr(attn, "value")
    assert hasattr(attn, "out_proj")
    assert hasattr(attn, "q_lin")
    assert hasattr(attn, "out_lin")


def test_spectra_v2_forward_and_shapes():
    cfg = make_config(hidden_size=64, num_heads=4, num_kv_heads=2)
    attn = SpectraV2Attention(cfg, pretrained_attn=DummyAttn(64, 4))
    attn.eval()

    for seq_len in [2, 8, 16, 25]:
        B, L, D = 2, seq_len, 64
        x = torch.randn(B, L, D)
        out, _ = attn(x)
        assert out.shape == (B, L, D), f"Expected shape {(B, L, D)}, got {out.shape}"
        assert not torch.isnan(out).any()


def test_spectra_v2_causal_mode():
    cfg = make_config(hidden_size=64, num_heads=4, num_kv_heads=2, is_decoder=True, model_type="gpt2")
    attn = SpectraV2Attention(cfg, is_decoder=True, pretrained_attn=DummyAttn(64, 4))
    attn.eval()

    B, L, D = 2, 16, 64
    x = torch.randn(B, L, D)
    out, _ = attn(x)
    assert out.shape == (B, L, D)
    assert not torch.isnan(out).any()


def test_spectra_v2_odd_dimension_and_rope():
    B, H, L, D_odd = 2, 4, 8, 17
    q = torch.randn(B, H, L, D_odd)
    k = torch.randn(B, H, L, D_odd)
    cos = torch.randn(1, 1, L, D_odd)
    sin = torch.randn(1, 1, L, D_odd)

    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
    assert q_rot.shape == (B, H, L, D_odd)
    assert k_rot.shape == (B, H, L, D_odd)
    assert not torch.isnan(q_rot).any()
    assert not torch.isnan(k_rot).any()


def test_spectra_v2_copy_weights_all_4_architectures():
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
    attn1 = SpectraV2Attention(cfg, pretrained_attn=distil_attn)
    assert torch.allclose(attn1.query.weight, distil_attn.q_lin.weight)
    assert torch.allclose(attn1.out_proj.weight, distil_attn.out_lin.weight)

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
    attn2 = SpectraV2Attention(cfg, pretrained_attn=roberta_attn)
    assert torch.allclose(attn2.query.weight, roberta_attn.self.query.weight)
    assert torch.allclose(attn2.out_proj.weight, roberta_attn.output.dense.weight)

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
    attn3 = SpectraV2Attention(cfg, pretrained_attn=gpt2_attn)
    assert torch.allclose(attn3.query.weight, gpt2_attn.c_attn.weight[:, :d].t())
    assert torch.allclose(attn3.out_proj.weight, gpt2_attn.c_proj.weight.t())

    # 4. Qwen / LLaMA style: q_proj, k_proj, v_proj, o_proj
    class QwenDummy(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(d, d)
            self.k_proj = nn.Linear(d, d)
            self.v_proj = nn.Linear(d, d)
            self.o_proj = nn.Linear(d, d)

    qwen_attn = QwenDummy()
    attn4 = SpectraV2Attention(cfg, pretrained_attn=qwen_attn)
    assert torch.allclose(attn4.query.weight, qwen_attn.q_proj.weight)
    assert torch.allclose(attn4.out_proj.weight, qwen_attn.o_proj.weight)


def test_spectra_v2_injection():
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
    res = inject_spectra_v2(layer=layer, model_config=cfg)
    assert res is not None
    assert isinstance(layer.attention, SpectraV2Attention)

    # Test whole-model injection
    class DummyTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = cfg
            self.transformer = nn.Module()
            self.transformer.h = nn.ModuleList([LayerDummy() for _ in range(2)])

    model = DummyTransformer()
    res_m = inject_spectra_v2(layer=model, model_config=cfg)
    assert res_m is not None
    for lyr in model.transformer.h:
        assert isinstance(lyr.attention, SpectraV2Attention)
