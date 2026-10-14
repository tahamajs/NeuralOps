"""Unit tests for Phase 1 UniversalArchitectureScanner."""

import torch
import torch.nn as nn
import pytest

from neuralops.surgery.scanner import (
    UniversalArchitectureScanner,
    Conv1DLinearAdapter,
    AttentionTargetSpec,
)


class MockBertSelfAttention(nn.Module):
    def __init__(self, d=64, heads=4):
        super().__init__()
        self.num_attention_heads = heads
        self.query = nn.Linear(d, d)
        self.key = nn.Linear(d, d)
        self.value = nn.Linear(d, d)

    def forward(self, x):
        return x


class MockBertSelfOutput(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.dense = nn.Linear(d, d)
        self.LayerNorm = nn.LayerNorm(d)

    def forward(self, hidden_states, input_tensor):
        return self.LayerNorm(self.dense(hidden_states) + input_tensor)


class MockBertAttention(nn.Module):
    def __init__(self, d=64, heads=4):
        super().__init__()
        self.self = MockBertSelfAttention(d, heads)
        self.output = MockBertSelfOutput(d)

    def forward(self, x):
        return self.output(self.self(x), x)


class MockLlamaAttention(nn.Module):
    def __init__(self, d=64, heads=4, kv_heads=2):
        super().__init__()
        self.num_heads = heads
        self.num_key_value_heads = kv_heads
        self.head_dim = d // heads
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d, kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)

    def forward(self, x):
        return x


class MockViTAttention(nn.Module):
    def __init__(self, d=64, heads=4):
        super().__init__()
        self.num_heads = heads
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)

    def forward(self, x):
        return x


class MockGPT2Conv1D(nn.Module):
    """Matches HuggingFace Conv1D."""
    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        self.nx = nx
        self.weight = nn.Parameter(torch.randn(nx, nf))
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        return x.view(size_out)


class MockGPT2Attention(nn.Module):
    def __init__(self, d=64, heads=4):
        super().__init__()
        self.num_heads = heads
        self.c_attn = MockGPT2Conv1D(3 * d, d)
        self.c_proj = MockGPT2Conv1D(d, d)

    def forward(self, x):
        return x


def test_scanner_bert_nested():
    bert_attn = MockBertAttention(d=64, heads=4)
    specs = UniversalArchitectureScanner.scan(bert_attn)
    assert len(specs) == 1
    spec = specs[0]
    assert spec.pattern_type == "bert_nested"
    assert spec.embed_dim == 64
    assert spec.num_heads == 4
    assert spec.attr_name == "self"
    assert spec.output_dense_module is not None
    assert spec.output_layernorm_module is not None
    W_q, b_q, W_k, b_k, W_v, b_v, W_o, b_o = spec.extracted_qkv
    assert W_q.shape == (64, 64)
    assert W_o.shape == (64, 64)


def test_scanner_llama_gqa():
    llama_attn = MockLlamaAttention(d=64, heads=4, kv_heads=2)
    specs = UniversalArchitectureScanner.scan(llama_attn)
    assert len(specs) == 1
    spec = specs[0]
    assert spec.pattern_type == "split_4proj"
    assert spec.num_heads == 4
    assert spec.num_kv_heads == 2
    assert spec.head_dim == 16
    assert spec.has_bias is False


def test_scanner_vit_fused_qkv():
    vit_attn = MockViTAttention(d=64, heads=4)
    specs = UniversalArchitectureScanner.scan(vit_attn)
    assert len(specs) == 1
    spec = specs[0]
    assert spec.pattern_type == "fused_qkv"
    assert spec.embed_dim == 64
    assert spec.num_heads == 4
    W_q, b_q, W_k, b_k, W_v, b_v, W_o, b_o = spec.extracted_qkv
    assert W_q.shape == (64, 64)
    assert W_k.shape == (64, 64)
    assert W_v.shape == (64, 64)


def test_scanner_gpt2_conv1d():
    gpt2_attn = MockGPT2Attention(d=64, heads=4)
    specs = UniversalArchitectureScanner.scan(gpt2_attn)
    assert len(specs) == 1
    spec = specs[0]
    assert spec.pattern_type == "conv1d_gpt2"
    assert spec.embed_dim == 64
    assert spec.num_heads == 4
    W_q, b_q, W_k, b_k, W_v, b_v, W_o, b_o = spec.extracted_qkv
    assert W_q.shape == (64, 64)
    assert W_o.shape == (64, 64)


def test_scanner_native_mha():
    mha = nn.MultiheadAttention(embed_dim=64, num_heads=4)
    specs = UniversalArchitectureScanner.scan(mha)
    assert len(specs) == 1
    spec = specs[0]
    assert spec.pattern_type == "native_mha"
    assert spec.embed_dim == 64
    assert spec.num_heads == 4
    W_q, b_q, W_k, b_k, W_v, b_v, W_o, b_o = spec.extracted_qkv
    assert W_q.shape == (64, 64)
    assert W_o.shape == (64, 64)
