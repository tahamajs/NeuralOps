# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from neuralops.attentions.adaptive_sparse_attention import (
    AdaptiveSparseAttention, LoRALinear, inject_adaptive_sparse)


def test_default_sparsity_per_family():
    cfg_enc = SimpleNamespace(model_type="bert", num_attention_heads=4, hidden_size=64, is_decoder=False)
    p_enc = AdaptiveSparseAttention(config=cfg_enc)
    assert p_enc.default_sparsity == pytest.approx(0.1, abs=1e-3)
    assert not p_enc.is_decoder

    cfg_dec = SimpleNamespace(model_type="gpt2", num_attention_heads=4, hidden_size=64, is_decoder=True)
    p_dec = AdaptiveSparseAttention(config=cfg_dec)
    assert p_dec.default_sparsity == pytest.approx(0.5, abs=1e-3)
    assert p_dec.is_decoder

    comp_cfg = SimpleNamespace(adaptive_sparsity_ratio=0.35)
    p_override = AdaptiveSparseAttention(config=cfg_enc, comp_config=comp_cfg)
    assert p_override.default_sparsity == pytest.approx(0.35, abs=1e-3)

    p_custom_enc = AdaptiveSparseAttention(config=cfg_enc, adaptive_sparse_encoder_sparsity=0.15)
    assert p_custom_enc.default_sparsity == pytest.approx(0.15, abs=1e-3)

    assert p_enc.sparsity_logits.requires_grad
    assert p_dec.sparsity_logits.requires_grad


def test_weight_copy_all_four_architectures():
    hidden_size = 64
    num_heads = 4
    head_dim = 16

    class MockDistilBertAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_lin = nn.Linear(hidden_size, hidden_size)
            self.k_lin = nn.Linear(hidden_size, hidden_size)
            self.v_lin = nn.Linear(hidden_size, hidden_size)
            self.out_lin = nn.Linear(hidden_size, hidden_size)

    d_src = MockDistilBertAttn()
    d_block = AdaptiveSparseAttention(d_model=hidden_size, num_heads=num_heads, pretrained_attn=d_src)
    for name, mod in [("q", d_block.q_proj), ("k", d_block.k_proj), ("v", d_block.v_proj), ("o", d_block.out_proj)]:
        assert float(torch.norm(mod.weight).item()) > 1e-4

    class MockRobertaSelf(nn.Module):
        def __init__(self):
            super().__init__()
            self.query = nn.Linear(hidden_size, hidden_size)
            self.key = nn.Linear(hidden_size, hidden_size)
            self.value = nn.Linear(hidden_size, hidden_size)

    class MockRobertaOutput(nn.Module):
        def __init__(self):
            super().__init__()
            self.dense = nn.Linear(hidden_size, hidden_size)

    class MockRobertaAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.self = MockRobertaSelf()
            self.output = MockRobertaOutput()

    r_src = MockRobertaAttn()
    r_block = AdaptiveSparseAttention(d_model=hidden_size, num_heads=num_heads, pretrained_attn=r_src)
    for name, mod in [("q", r_block.q_proj), ("k", r_block.k_proj), ("v", r_block.v_proj), ("o", r_block.out_proj)]:
        assert float(torch.norm(mod.weight).item()) > 1e-4

    class MockConv1D(nn.Module):
        def __init__(self, in_f, out_f):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(in_f, out_f))
            self.bias = nn.Parameter(torch.zeros(out_f))

    class MockGPT2Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.c_attn = MockConv1D(hidden_size, 3 * hidden_size)
            self.c_proj = MockConv1D(hidden_size, hidden_size)

    g_src = MockGPT2Attn()
    g_block = AdaptiveSparseAttention(d_model=hidden_size, num_heads=num_heads, pretrained_attn=g_src)
    for name, mod in [("q", g_block.q_proj), ("k", g_block.k_proj), ("v", g_block.v_proj), ("o", g_block.out_proj)]:
        assert float(torch.norm(mod.weight).item()) > 1e-4

    class MockQwenAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.head_dim = head_dim
            self.num_heads = num_heads
            self.num_key_value_heads = 2
            self.q_proj = nn.Linear(hidden_size, num_heads * head_dim)
            self.k_proj = nn.Linear(hidden_size, 2 * head_dim)
            self.v_proj = nn.Linear(hidden_size, 2 * head_dim)
            self.o_proj = nn.Linear(num_heads * head_dim, hidden_size)

    q_src = MockQwenAttn()
    cfg_qwen = SimpleNamespace(model_type="qwen2", num_attention_heads=num_heads, num_key_value_heads=2, hidden_size=hidden_size)
    q_block = AdaptiveSparseAttention(config=cfg_qwen, pretrained_attn=q_src)
    assert q_block.num_kv_heads == 2
    assert q_block.kv_repeat == 2
    for name, mod in [("q", q_block.q_proj), ("k", q_block.k_proj), ("v", q_block.v_proj), ("o", q_block.out_proj)]:
        assert float(torch.norm(mod.weight).item()) > 1e-4


def test_lora_freezing():
    class DummyAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(64, 64)
            self.k_proj = nn.Linear(64, 64)
            self.v_proj = nn.Linear(64, 64)
            self.out_proj = nn.Linear(64, 64)

    old = DummyAttn()
    p = AdaptiveSparseAttention(d_model=64, num_heads=4, pretrained_attn=old, lora_rank=16)
    for mod in (p.q_proj, p.k_proj, p.v_proj, p.out_proj):
        assert not mod.weight.requires_grad
        if mod.bias is not None:
            assert not mod.bias.requires_grad
        assert mod.lora_A.requires_grad
        assert mod.lora_B.requires_grad


def test_rope_fallback_both_signatures():
    class MockRotarySeqLen(nn.Module):
        def forward(self, x, seq_len=None):
            B, H, N, d = x.shape
            cos = torch.ones(B, 1, N, d)
            sin = torch.zeros(B, 1, N, d)
            return cos, sin

    class MockRotaryPosIds(nn.Module):
        def forward(self, x, position_ids=None):
            B, H, N, d = x.shape
            cos = torch.ones(B, 1, N, d)
            sin = torch.zeros(B, 1, N, d)
            return cos, sin

    p1 = AdaptiveSparseAttention(d_model=64, num_heads=4)
    p1.rotary_emb = MockRotarySeqLen()
    x = torch.randn(2, 8, 64)
    out1, _ = p1(x, position_embeddings=None)
    assert out1.shape == (2, 8, 64)
    assert torch.isfinite(out1).all()

    p2 = AdaptiveSparseAttention(d_model=64, num_heads=4)
    p2.rotary_emb = MockRotaryPosIds()
    out2, _ = p2(x, position_embeddings=None)
    assert out2.shape == (2, 8, 64)
    assert torch.isfinite(out2).all()


def test_causal_mask_before_topk():
    p = AdaptiveSparseAttention(d_model=64, num_heads=4, is_decoder=True, is_causal=True, sparsity_ratio=0.5)
    L = 6
    x = torch.randn(1, L, 64)
    _, attn = p(x, output_attentions=True)
    assert attn is not None
    for i in range(L):
        for j in range(i + 1, L):
            assert torch.all(attn[:, :, i, j] == 0.0), f"Future position ({i}, {j}) has non-zero attention {attn[:, :, i, j]}"


def test_topk_threshold_correctness():
    sparsity = 0.5
    L = 10
    H = 4
    p = AdaptiveSparseAttention(d_model=64, num_heads=H, is_decoder=False, is_causal=False, sparsity_ratio=sparsity)
    p.eval()
    x = torch.randn(2, L, 64)
    _, attn = p(x, output_attentions=True)
    assert attn is not None
    expected_k = int(round((1.0 - sparsity) * L))
    topk_mass = torch.topk(attn, expected_k, dim=-1).values.sum(dim=-1)
    assert torch.all(topk_mass > 0.65)
    assert (attn < 0.05).float().mean().item() > 0.25


def test_head_mask_shapes():
    p = AdaptiveSparseAttention(d_model=64, num_heads=4)
    x = torch.randn(2, 8, 64)
    H = 4
    B = 2

    hm_1d = torch.ones(H)
    out1, _ = p(x, head_mask=hm_1d)
    assert out1.shape == (B, 8, 64)

    hm_2d = torch.ones(B, H)
    out2, _ = p(x, head_mask=hm_2d)
    assert out2.shape == (B, 8, 64)

    hm_4d = torch.ones(B, H, 1, 1)
    out4, _ = p(x, head_mask=hm_4d)
    assert out4.shape == (B, 8, 64)


def test_nested_encoder_bypass():
    class MockBertSelfAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.query = nn.Linear(64, 64)
            self.key = nn.Linear(64, 64)
            self.value = nn.Linear(64, 64)

    src = MockBertSelfAttention()
    p = AdaptiveSparseAttention(d_model=64, num_heads=4, pretrained_attn=src)
    assert p.is_nested_encoder is True

    x = torch.randn(2, 8, 64)
    out, _ = p(x)
    assert out.shape == (2, 8, 64)


def test_injection_safety():
    class MissingProjectionsLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = nn.Linear(64, 64)

    layer = MissingProjectionsLayer()
    res = inject_adaptive_sparse(layer=layer)
    assert res is None
