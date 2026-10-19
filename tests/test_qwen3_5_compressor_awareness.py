#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Comprehensive test suite verifying architecture-aware compression, dynamic rank selection,
recovery parameter selection, and LoRA preservation on Qwen3.5 and across model families.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from neuralops.compression_engine.a3_compressor import A3Compressor
from neuralops.compression_engine.aa_svd_compressor import AASVDCompressor
from neuralops.compression_engine.cur_moe_compressor import CURMoECompressor
from neuralops.compression_engine.direct_svd import DirectSVDCompressor
from neuralops.compression_engine.dynamic_rank import \
    DynamicEntropySVDCompressor
from neuralops.core.registry import (COMPRESSOR_REGISTRY,
                                     is_module_declared_compatible)
from neuralops.utils.compat import (get_dynamic_compression_rank,
                                    get_recovery_params, is_decoder_family,
                                    should_compress_linear)
from neuralops.utils.lora import apply_lora
from neuralops.utils.model_family import ModelAdapter, detect_family


class DummyQwen35Block(nn.Module):
    """Synthetic Qwen3.5 hybrid block containing DeltaNet, GQA, MoE/FFN, and Norms."""

    def __init__(self, hidden_size: int = 1024, ffn_dim: int = 3584):
        super().__init__()
        self.delta_net = nn.Module()
        self.delta_net.in_proj_qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.delta_net.out_proj = nn.Linear(hidden_size, hidden_size)

        self.attn = nn.Module()
        self.attn.q_proj = nn.Linear(hidden_size, hidden_size)
        self.attn.k_proj = nn.Linear(hidden_size, 256)
        self.attn.v_proj = nn.Linear(hidden_size, 256)
        self.attn.o_proj = nn.Linear(hidden_size, hidden_size)

        self.expert_router = nn.Linear(hidden_size, 4)

        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(hidden_size, ffn_dim)
        self.mlp.up_proj = nn.Linear(hidden_size, ffn_dim)
        self.mlp.down_proj = nn.Linear(ffn_dim, hidden_size)


class DummyQwen35Model(nn.Module):
    """Full dummy Qwen3.5 architecture."""

    def __init__(self, hidden_size: int = 1024, ffn_dim: int = 3584, num_layers: int = 2):
        super().__init__()
        self.config = SimpleNamespace(
            model_type="qwen3_5",
            hidden_size=hidden_size,
            num_attention_heads=8,
            num_key_value_heads=2,
            head_dim=256,
            num_hidden_layers=num_layers,
            intermediate_size=ffn_dim,
        )
        self.layers = nn.ModuleList([DummyQwen35Block(hidden_size, ffn_dim) for _ in range(num_layers)])
        self.score = nn.Linear(hidden_size, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            h = layer.mlp.down_proj(torch.relu(layer.mlp.up_proj(x)))
            x = x + h
        return self.score(x)


class DummyStandardTransformer(nn.Module):
    """Standard BERT / RoBERTa / GPT model."""

    def __init__(self, family: str = "bert", hidden_size: int = 64):
        super().__init__()
        self.config = SimpleNamespace(
            model_type=family,
            hidden_size=hidden_size,
            num_attention_heads=4,
            num_hidden_layers=1,
            intermediate_size=hidden_size * 4,
        )
        self.attn = nn.Module()
        self.attn.q_proj = nn.Linear(hidden_size, hidden_size)
        self.attn.k_proj = nn.Linear(hidden_size, hidden_size)
        self.attn.v_proj = nn.Linear(hidden_size, hidden_size)
        self.attn.out_proj = nn.Linear(hidden_size, hidden_size)
        self.mlp = nn.Module()
        self.mlp.fc1 = nn.Linear(hidden_size, hidden_size * 4)
        self.mlp.fc2 = nn.Linear(hidden_size * 4, hidden_size)
        self.classifier = nn.Linear(hidden_size, 2)


def test_qwen3_5_architecture_aware_layer_filtering():
    """Verify should_compress_linear correctly protects sensitive layers on Qwen3.5 while allowing FFN."""
    model = DummyQwen35Model()
    assert detect_family(model) == "qwen3_5"

    assert not should_compress_linear(model, "layers.0.delta_net.in_proj_qkv")
    assert not should_compress_linear(model, "layers.0.delta_net.out_proj")
    assert not should_compress_linear(model, "layers.0.attn.q_proj")
    assert not should_compress_linear(model, "layers.0.attn.k_proj")
    assert not should_compress_linear(model, "layers.0.attn.v_proj")
    assert not should_compress_linear(model, "layers.0.attn.o_proj")
    assert not should_compress_linear(model, "layers.0.expert_router")
    assert not should_compress_linear(model, "layers.0.mlp.gate_proj")
    assert not should_compress_linear(model, "score")

    assert should_compress_linear(model, "layers.0.mlp.up_proj")
    assert should_compress_linear(model, "layers.0.mlp.down_proj")


def test_standard_family_layer_filtering_preserved():
    """Verify standard models (BERT, RoBERTa, GPT) preserve their original compression filtering."""
    bert = DummyStandardTransformer("bert", hidden_size=64)
    gpt = DummyStandardTransformer("gpt2", hidden_size=64)

    assert should_compress_linear(bert, "attn.q_proj")
    assert should_compress_linear(bert, "mlp.fc1")
    assert not should_compress_linear(bert, "classifier")

    assert should_compress_linear(gpt, "attn.q_proj")
    assert should_compress_linear(gpt, "mlp.fc1")
    assert not should_compress_linear(gpt, "classifier")


def test_compressors_skip_qwen3_5_sensitive_layers():
    """Verify that actual compressor instances skip sensitive layers on Qwen3.5."""
    model = DummyQwen35Model(hidden_size=128, ffn_dim=256, num_layers=1)

    comp = DirectSVDCompressor(model, rank=32)
    comp.compress()
    assert isinstance(model.layers[0].delta_net.in_proj_qkv, nn.Linear)
    assert isinstance(model.layers[0].attn.q_proj, nn.Linear)
    assert isinstance(model.layers[0].expert_router, nn.Linear)
    assert not isinstance(model.layers[0].mlp.down_proj, nn.Linear) or hasattr(model.layers[0].mlp.down_proj, "U")


def test_dynamic_rank_selection():
    """Verify dynamic rank computes min(256, max(128, min(in_f, out_f) // 4)) for Qwen3.5."""
    rank_qwen = get_dynamic_compression_rank(1024, 3584, model_or_family="qwen3_5", default_rank=16)
    assert rank_qwen == 256

    rank_qwen_small = get_dynamic_compression_rank(400, 400, model_or_family="qwen3_5", default_rank=16)
    assert rank_qwen_small == 128

    rank_bert = get_dynamic_compression_rank(768, 3072, model_or_family="bert", default_rank=32)
    assert rank_bert == 32

    rank_override = get_dynamic_compression_rank(1024, 3584, model_or_family="qwen3_5", comp_config={"qwen3_5_rank": 64})
    assert rank_override == 64


def test_get_recovery_params():
    """Verify get_recovery_params unfreezes only LoRA adapters and classification heads, freezing all else."""
    model = DummyQwen35Model(hidden_size=64, ffn_dim=128, num_layers=1)
    model, _ = apply_lora(model, r=8, alpha=16.0, target_modules={"down_proj"})

    model.layers[0].mlp.up_proj.U = nn.Parameter(torch.randn(128, 16))
    model.layers[0].mlp.up_proj.V = nn.Parameter(torch.randn(16, 64))

    trainable = get_recovery_params(model)
    trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]

    assert any("lora_" in n for n in trainable_names)
    assert any("score" in n for n in trainable_names)
    assert not any(n.endswith(".U") for n in trainable_names)
    assert not any(n.endswith(".V") for n in trainable_names)
    assert not any("mlp.up_proj.weight" in n for n in trainable_names)


def test_lora_preservation_and_gradient_flow():
    """Verify that LoRA adapters on frozen backbones receive non-zero gradients after compressor injection."""
    model = DummyQwen35Model(hidden_size=64, ffn_dim=128, num_layers=1)
    model, num_lora = apply_lora(model, r=8, alpha=16.0, target_modules={"down_proj"})
    assert num_lora > 0

    get_recovery_params(model)

    x = torch.randn(2, 64)
    out = model(x)
    loss = out.sum()
    loss.backward()

    lora_grads = [p.grad for name, p in model.named_parameters() if "lora_" in name and p.grad is not None]
    assert len(lora_grads) > 0
    assert any(g.abs().sum() > 0 for g in lora_grads)
    assert all(torch.isfinite(g).all() for g in lora_grads)


def test_qwen3_5_capabilities_and_compatibility():
    """Verify capability flags for Qwen3.5 hybrid architecture in registry."""
    adapter = ModelAdapter(DummyQwen35Model())
    assert adapter.causal is True
    assert adapter.supports_gqa is True
    assert adapter.supports_delta_net is True
    assert adapter.supports_moe is True

    compat, _ = is_module_declared_compatible("adaptive_svd", "qwen3_5")
    assert compat is True
