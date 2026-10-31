#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors

from .testing_utils import MockConfig, DummyDistilBertAttention
import pytest
import torch
import torch.nn as nn
from neuralops.attentions.svd_linear_decay import (
    DecayDecayLinearAttention,
    SVDLinearDecayAttention,
    inject_svd_linear_decay,
    inject_svd_linear_decay_attention,
)

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

def test_svd_linear_decay_basic():
    cfg = MockConfig()
    attn = DecayDecayLinearAttention(config=cfg, rank=8)
    assert attn.use_decay_gating is True
    x = torch.randn(2, 8, 64)
    out, _ = attn(x)
    assert out.shape == (2, 8, 64)

def test_svd_linear_decay_trainable_lora_weights():
    cfg = MockConfig(lora_rank=8)
    attn = DecayDecayLinearAttention(config=cfg, rank=8, freeze_svd=True)
    assert attn.q_proj.V.requires_grad is False
    assert attn.q_proj.U.requires_grad is False
    assert attn.q_proj.lora_A.requires_grad is True
    assert attn.q_proj.lora_B.requires_grad is True

def test_svd_linear_decay_distilbert_weight_copy():
    cfg = MockConfig()
    old_attn = DummyDistilBertAttention(64)
    attn = DecayDecayLinearAttention(config=cfg, rank=8, old_attn=old_attn)
    x = torch.randn(2, 8, 64)
    out, _ = attn(x)
    assert out.shape == (2, 8, 64)

def test_svd_linear_decay_roberta_weight_copy():
    cfg = MockConfig()
    old_attn = DummyRobertaAttention(64)
    attn = DecayDecayLinearAttention(config=cfg, rank=8, old_attn=old_attn)
    assert attn.is_nested_encoder
    x = torch.randn(2, 8, 64)
    out, _ = attn(x)
    assert out.shape == (2, 8, 64)

def test_svd_linear_decay_gpt2_weight_copy():
    cfg = MockConfig(is_decoder=True, model_type="gpt2")
    old_attn = DummyGPT2Attention(64)
    attn = DecayDecayLinearAttention(config=cfg, rank=8, old_attn=old_attn)
    assert attn.is_decoder
    x = torch.randn(2, 8, 64)
    out, _ = attn(x)
    assert out.shape == (2, 8, 64)

def test_svd_linear_decay_qwen_gqa_and_rope():
    cfg = MockConfig(hidden_size=64, num_attention_heads=4, num_key_value_heads=2, head_dim=16, is_decoder=True, model_type="qwen2")
    old_attn = DummyQwenAttention(dim=64, num_kv_heads=2, head_dim=16)
    attn = DecayDecayLinearAttention(config=cfg, rank=8, old_attn=old_attn)
    assert attn.kv_repeat == 2
    x = torch.randn(2, 8, 64)
    cos = torch.randn(2, 8, 16)
    sin = torch.randn(2, 8, 16)
    out, _ = attn(x, position_embeddings=(cos, sin))
    assert out.shape == (2, 8, 64)

def test_svd_linear_decay_inject_safe():
    class DummyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = DummyDistilBertAttention(64)

    layer = DummyLayer()
    res = inject_svd_linear_decay(layer=layer, model_config=MockConfig(), old_attn=layer.attention, rank=8)
    assert res is not None
    x = torch.randn(2, 8, 64)
    out, _ = layer.attention(x)
    assert out.shape == (2, 8, 64)


def test_svd_linear_decay_alias_and_mode():
    cfg = MockConfig(hidden_size=64, num_attention_heads=4)
    attn = SVDLinearDecayAttention(config=cfg, rank=8)
    assert attn.use_decay_gating is True
    assert attn.mode in ("decay_linear", "linear_decay", "decay", "gated_linear")
    x = torch.randn(2, 16, 64)
    out, _ = attn(x)
    assert out.shape == (2, 16, 64)
    assert torch.isfinite(out).all()


def test_svd_linear_decay_linear_complexity_long_seq():
    cfg = MockConfig(hidden_size=64, num_attention_heads=4, chunk_size=32)
    attn = DecayDecayLinearAttention(config=cfg, rank=8)
    # Long sequence N=256 processed in linear O(N) chunked time without N x N dense matrix
    x = torch.randn(1, 256, 64)
    out, _ = attn(x)
    assert out.shape == (1, 256, 64)
    assert torch.isfinite(out).all()


def test_svd_linear_decay_step_recurrent_parity():
    cfg = MockConfig(hidden_size=64, num_attention_heads=4)
    attn = DecayDecayLinearAttention(config=cfg, rank=8)
    attn.eval()
    x = torch.randn(1, 4, 64)
    with torch.no_grad():
        out_seq, _, (S_final, Z_final) = attn(hidden_states=x, return_state=True)
        state = None
        out_steps = []
        for t in range(4):
            x_t = x[:, t : t + 1, :]
            out_t, state = attn.step_recurrent(x_t, past_state=state)
            out_steps.append(out_t)
        out_streamed = torch.cat(out_steps, dim=1)
        max_diff = torch.max(torch.abs(out_seq - out_streamed)).item()
        assert max_diff < 1e-4, f"Recurrent divergence: {max_diff}"

