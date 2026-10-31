#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
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


class MockLoRALinear(nn.Module):
    """Simulates a peft.tuners.lora.Linear projection."""
    def __init__(self, in_features=64, out_features=64, r=4, scaling=2.0):
        super().__init__()
        self.base_layer = nn.Linear(in_features, out_features, bias=False)
        nn.init.constant_(self.base_layer.weight, 1.0)
        self.lora_A = nn.ModuleDict({"default": nn.Linear(in_features, r, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(r, out_features, bias=False)})
        nn.init.constant_(self.lora_A["default"].weight, 0.5)
        nn.init.constant_(self.lora_B["default"].weight, 0.5)
        self.scaling = {"default": scaling}
        self.merged = False

    @property
    def weight(self):
        return self.base_layer.weight

    @property
    def bias(self):
        return self.base_layer.bias


def test_scanner_lora_merge_and_extraction():
    """Verify UniversalArchitectureScanner does not strip LoRA adapters and merges delta weights."""
    llama_attn = MockLlamaAttention(d=64, heads=4, kv_heads=4)
    # Wrap q_proj with LoRA
    llama_attn.q_proj = MockLoRALinear(in_features=64, out_features=64, r=4, scaling=2.0)

    specs = UniversalArchitectureScanner.scan(llama_attn)
    assert len(specs) == 1
    spec = specs[0]
    assert spec.extra_metadata is not None
    assert "lora_adapters" in spec.extra_metadata
    assert "q" in spec.extra_metadata["lora_adapters"]

    W_q, _, _, _, _, _, _, _ = spec.extracted_qkv
    # Base weight is 1.0. Delta is B @ A * scaling = (0.5 * 0.5 * 4 rank) * 2.0 = 1.0 * 2.0 = 2.0
    # Expected merged weight is 1.0 + 2.0 = 3.0
    expected_w = 1.0 + (4 * 0.25 * 2.0)
    assert torch.allclose(W_q, torch.tensor(expected_w))


def test_scanner_prefix_pruning():
    """Verify scanner prunes children of matched attention blocks."""
    llama_attn = MockLlamaAttention(d=64, heads=4, kv_heads=4)
    model = nn.Sequential(llama_attn)
    specs = UniversalArchitectureScanner.scan(model)
    # Exactly 1 spec found, children q_proj/k_proj are not separate attention targets
    assert len(specs) == 1


def test_scanner_root_level_prefix_pruning():
    """Verify direct scan of root attention block does not inspect internal subprojections."""
    llama_attn = MockLlamaAttention(d=64, heads=4, kv_heads=4)
    specs = UniversalArchitectureScanner.scan(llama_attn)
    assert len(specs) == 1
    assert specs[0].attr_name == ""


class MockFusedGQAAttention(nn.Module):
    """Fused QKV attention with GQA where total_out % 3 == 0 (16 Q heads, 4 KV heads = 24 total heads)."""

    def __init__(self, d=1024, heads=16, kv_heads=4):
        super().__init__()
        self.num_heads = heads
        self.num_key_value_heads = kv_heads
        self.head_dim = d // heads  # 64
        total_qkv_out = (heads + 2 * kv_heads) * self.head_dim  # (16 + 8) * 64 = 1536
        self.qkv = nn.Linear(d, total_qkv_out, bias=False)
        self.proj = nn.Linear(d, d, bias=False)

    def forward(self, x):
        return x


def test_scanner_fused_qkv_gqa_arithmetic():
    """Verify GQA with fused QKV divisible by 3 (1536 % 3 == 0) correctly isolates Q, K, V."""
    gqa_attn = MockFusedGQAAttention(d=1024, heads=16, kv_heads=4)
    specs = UniversalArchitectureScanner.scan(gqa_attn)
    assert len(specs) == 1
    spec = specs[0]
    assert spec.pattern_type == "fused_qkv"
    assert spec.num_heads == 16
    assert spec.num_kv_heads == 4
    assert spec.head_dim == 64

    W_q, _, W_k, _, W_v, _, W_o, _ = spec.extracted_qkv
    # Q: 16 heads * 64 = 1024
    assert W_q.shape == (1024, 1024)
    # K: 4 heads * 64 = 256
    assert W_k.shape == (256, 1024)
    # V: 4 heads * 64 = 256
    assert W_v.shape == (256, 1024)
    assert W_o.shape == (1024, 1024)


class MockDoRALinear(nn.Module):
    """Simulates a peft.tuners.lora.dora.Linear projection with magnitude vector."""

    def __init__(self, in_features=64, out_features=64, r=4, scaling=2.0):
        super().__init__()
        self.base_layer = nn.Linear(in_features, out_features, bias=False)
        nn.init.constant_(self.base_layer.weight, 1.0)
        self.lora_A = nn.ModuleDict({"default": nn.Linear(in_features, r, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(r, out_features, bias=False)})
        nn.init.constant_(self.lora_A["default"].weight, 0.5)
        nn.init.constant_(self.lora_B["default"].weight, 0.5)
        self.scaling = {"default": scaling}
        # Learned magnitude vector: all 2.0
        self.lora_magnitude_vector = nn.ParameterDict(
            {"default": nn.Parameter(torch.full((out_features, 1), 2.0))}
        )
        self.merged = False


def test_scanner_dora_magnitude_normalization():
    """Verify DoRA magnitude vector normalization is applied during weight extraction."""
    llama_attn = MockLlamaAttention(d=64, heads=4, kv_heads=4)
    llama_attn.q_proj = MockDoRALinear(in_features=64, out_features=64, r=4, scaling=2.0)
    specs = UniversalArchitectureScanner.scan(llama_attn)
    assert len(specs) == 1
    W_q, _, _, _, _, _, _, _ = specs[0].extracted_qkv

    # Raw V has all elements = 1.0 + (4 * 0.25 * 2.0) = 3.0
    # Column norm of row of 64 elements of 3.0: sqrt(64 * 9.0) = sqrt(576) = 24.0
    # Normalized: 3.0 / 24.0 = 0.125
    # Rescaled by magnitude 2.0: 2.0 * 0.125 = 0.25
    expected_elem = 2.0 * (3.0 / 24.0)
    assert torch.allclose(W_q, torch.tensor(expected_elem))

