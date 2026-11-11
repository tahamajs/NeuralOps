#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for unified modular surgical delta serialization, loading, and Model Cards."""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file

from neuralops.hub import generate_unified_model_card, save_surgical_delta
from neuralops.models.surgical_io import extract_surgical_manifest


class _MockSurgeriedAttn(nn.Module):
    is_custom_attention = True
    target_attention = "alibi"

    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.embed_dim = dim
        self.num_heads = 2
        self.q_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.o_proj(self.q_proj(x))


class _MockAnchorAttn(nn.Module):
    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)


class _MockCompressedLinear(nn.Module):
    is_compressed = True

    def __init__(self, in_f: int = 16, out_f: int = 16, rank: int = 4) -> None:
        super().__init__()
        self.in_features = in_f
        self.out_features = out_f
        self.rank = rank
        self.U = nn.Parameter(torch.randn(out_f, rank))
        self.V = nn.Parameter(torch.randn(rank, in_f))
        self.bias = nn.Parameter(torch.zeros(out_f))


class _MockVLMModel(nn.Module):
    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.vision_tower = nn.Module()
        self.vision_tower.self_attn = nn.Linear(dim, dim)
        self.multi_modal_projector = nn.Linear(dim, dim)
        self.language_model = nn.Module()
        self.language_model.model = nn.Module()
        self.language_model.model.embed_tokens = nn.Embedding(32, dim)
        layer0 = nn.Module()
        layer0.self_attn = _MockSurgeriedAttn(dim)
        layer0.input_layernorm = nn.LayerNorm(dim)
        layer0.mlp = _MockCompressedLinear(dim, dim, rank=4)
        layer1 = nn.Module()
        layer1.self_attn = _MockAnchorAttn(dim)
        layer1.input_layernorm = nn.LayerNorm(dim)
        self.language_model.model.layers = nn.ModuleList([layer0, layer1])


def test_save_surgical_delta_excludes_frozen_vision_and_anchors(tmp_path: Path) -> None:
    model = _MockVLMModel()
    for p in model.parameters():
        p.requires_grad = False
    for p in model.language_model.model.layers[0].self_attn.parameters():
        p.requires_grad = True
    for p in model.language_model.model.layers[0].input_layernorm.parameters():
        p.requires_grad = True
    for p in model.language_model.model.layers[0].mlp.parameters():
        p.requires_grad = True

    stale = tmp_path / "pytorch_model.bin"
    stale.write_bytes(b"legacy-full-weights")

    trained_names = [n for n, p in model.named_parameters() if p.requires_grad]
    summary = save_surgical_delta(
        model,
        tmp_path,
        base_model_id="llava-hf/llava-interleave-qwen-0.5b-hf",
        surgeried_prefixes=["language_model.model.layers.0.self_attn"],
        trained_param_names=trained_names,
        anchor_prefixes=["language_model.model.layers.1.self_attn"],
        operations=[{"component_type": "attention", "operator_key": "alibi", "pattern": "hybrid_3_1"}],
    )

    assert not stale.exists(), "Legacy full weights must be purged"
    weights_file = tmp_path / "attention_weights.safetensors"
    manifest_file = tmp_path / "surgical_manifest.json"
    assert weights_file.exists()
    assert manifest_file.exists()

    saved = load_file(str(weights_file))
    assert any(k.startswith("language_model.model.layers.0.self_attn.") for k in saved)
    assert any(k.startswith("language_model.model.layers.0.mlp.") for k in saved)
    assert any(k.startswith("language_model.model.layers.0.input_layernorm.") for k in saved)
    assert not any(k.startswith("vision_tower.") for k in saved), "Frozen vision_tower leaked!"
    assert not any(k.startswith("language_model.model.layers.1.self_attn.") for k in saved), "Anchor leaked!"
    assert not any("embed_tokens" in k for k in saved), "Frozen embed_tokens leaked!"

    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "neuralops.surgery/v2"
    assert summary["saved_tensor_count"] == len(saved)
    extracted = extract_surgical_manifest(model)
    assert "language_model.model.layers.0.self_attn" in extracted["submodules"]
    assert "language_model.model.layers.0.mlp" in extracted["submodules"]


def test_unified_model_card_all_six_zoos_honest_audit(tmp_path: Path) -> None:
    (tmp_path / "attention_weights.safetensors").write_bytes(b"0" * 4096)
    (tmp_path / "config.json").write_text('{"attention_type": "cosformer"}', encoding="utf-8")

    for zoo_type in (
        "bert_attention",
        "bert_compressor",
        "qwen_attention",
        "qwen_compressor",
        "llava_vlm",
        "tinyllama",
    ):
        card = generate_unified_model_card(
            zoo_type=zoo_type,
            mechanism_key="cosformer",
            repo_id=f"tahamajs/test-{zoo_type}",
            out_dir=tmp_path,
            metrics={
                "acc_before": 40.0,
                "acc_after": 62.5,
                "ppl_before": 120.0,
                "ppl_after": 22.5,
                "orig_params": 100_000_000,
                "comp_params": 50_000_000,
                "steps_s1": 20,
                "steps_s2": 40,
                "train_secs": 30.0,
                "peak_vram_mb": 12000,
            },
        )
        assert "license: mit" in card
        assert "model-index:" in card
        assert "load_surgeried_model" in card
        assert "from scripts.benchmarks" not in card
        assert "attention_weights.safetensors" in card
        assert "pytorch_model.bin" not in card
