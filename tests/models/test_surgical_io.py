#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Regression tests for surgical model I/O and checkpoint safety."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

from neuralops.models.surgical_io import (
    extract_surgical_manifest,
    load_surgical_checkpoint,
    save_surgical_delta,
)


class DummyAttention(nn.Module):
    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.is_custom_attention = True
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_proj(self.q_proj(x))


class DummyModel(nn.Module):
    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.embed = nn.Embedding(32, dim)
        self.attn = DummyAttention(dim)
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        h = self.attn(h)
        h = self.norm(h)
        return self.mlp(h)


def test_save_surgical_delta_preserves_existing_user_files(tmp_path: Path) -> None:
    """P0-1 Regression: save_surgical_delta must NOT delete user files by default."""
    model = DummyModel()
    for p in model.parameters():
        p.requires_grad = False
    for p in model.attn.parameters():
        p.requires_grad = True

    user_full_weights = tmp_path / "model.safetensors"
    user_full_weights.write_bytes(b"valuable-base-model-weights")
    user_optimizer = tmp_path / "optimizer.pt"
    user_optimizer.write_bytes(b"valuable-optimizer-state")
    user_bin = tmp_path / "pytorch_model.bin"
    user_bin.write_bytes(b"valuable-legacy-bin")

    trained_names = [n for n, p in model.named_parameters() if p.requires_grad]
    summary = save_surgical_delta(
        model,
        tmp_path,
        surgeried_prefixes=["attn"],
        trained_param_names=trained_names,
    )

    assert summary["saved_tensor_count"] > 0
    assert (tmp_path / "attention_weights.safetensors").exists()
    assert (tmp_path / "surgical_manifest.json").exists()

    # User files must be completely untouched
    assert user_full_weights.exists()
    assert user_full_weights.read_bytes() == b"valuable-base-model-weights"
    assert user_optimizer.exists()
    assert user_optimizer.read_bytes() == b"valuable-optimizer-state"
    assert user_bin.exists()
    assert user_bin.read_bytes() == b"valuable-legacy-bin"


def test_save_surgical_delta_purges_only_when_explicitly_requested(tmp_path: Path) -> None:
    """P0-1 Regression: remove_full_weights=True only purges when caller explicitly opts in."""
    model = DummyModel()
    for p in model.parameters():
        p.requires_grad = False
    for p in model.attn.parameters():
        p.requires_grad = True

    stale_safetensors = tmp_path / "model.safetensors"
    stale_safetensors.write_bytes(b"stale")
    stale_opt = tmp_path / "optimizer.pt"
    stale_opt.write_bytes(b"stale")

    save_surgical_delta(
        model,
        tmp_path,
        surgeried_prefixes=["attn"],
        remove_full_weights=True,
    )

    assert not stale_safetensors.exists()
    assert not stale_opt.exists()
    assert (tmp_path / "attention_weights.safetensors").exists()


def test_load_surgical_checkpoint_raises_on_unexpected_keys_when_strict(tmp_path: Path) -> None:
    """P1 Regression: Strict load must raise RuntimeError on unexpected keys rather than silently skipping."""
    model = DummyModel()
    manifest = extract_surgical_manifest(model)
    manifest["schema_version"] = "neuralops.surgery/v2"
    manifest["weights_file"] = "attention_weights.safetensors"

    (tmp_path / "surgical_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    # Write a checkpoint containing an unrecognized tensor key
    bad_state = {
        "nonexistent_layer.weight": torch.randn(16, 16),
        "attn.q_proj.weight": model.attn.q_proj.weight.clone(),
    }
    save_file(bad_state, str(tmp_path / "attention_weights.safetensors"))

    with pytest.raises(RuntimeError, match="unexpected keys"):
        load_surgical_checkpoint(model, tmp_path, strict=True)


def test_load_surgical_checkpoint_raises_on_missing_surgical_keys_when_strict(tmp_path: Path) -> None:
    """P1 Regression: Strict load must raise RuntimeError if expected surgical parameters are missing."""
    model = DummyModel()
    manifest = extract_surgical_manifest(model)
    manifest["schema_version"] = "neuralops.surgery/v2"
    manifest["weights_file"] = "attention_weights.safetensors"
    manifest["surgeried_prefixes"] = ["attn"]

    (tmp_path / "surgical_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    # Omit attn.k_proj.weight and attn.v_proj.weight
    partial_state = {
        "attn.q_proj.weight": model.attn.q_proj.weight.clone(),
    }
    save_file(partial_state, str(tmp_path / "attention_weights.safetensors"))

    with pytest.raises(RuntimeError, match="expected surgical parameters were missing"):
        load_surgical_checkpoint(model, tmp_path, strict=True)


def test_load_surgical_checkpoint_raises_on_missing_lora_target_module(tmp_path: Path) -> None:
    """P1 Regression: Strict load must raise KeyError if LoRA target module path does not exist."""
    model = DummyModel()
    manifest = {
        "schema_version": "neuralops.surgery/v2",
        "weights_file": "attention_weights.safetensors",
        "submodules": {
            "nonexistent_submodule": {
                "category": "lora_adapter",
                "class_name": "SimpleLoRALinear",
                "rank": 4,
            }
        },
    }
    (tmp_path / "surgical_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    save_file({}, str(tmp_path / "attention_weights.safetensors"))

    with pytest.raises(KeyError, match="Target module path 'nonexistent_submodule'"):
        load_surgical_checkpoint(model, tmp_path, strict=True)


def test_load_surgical_checkpoint_raises_on_zero_loaded_keys(tmp_path: Path) -> None:
    """P1 Regression: Checkpoint load must raise RuntimeError if 0 tensors were loaded."""
    model = DummyModel()
    manifest = {
        "schema_version": "neuralops.surgery/v2",
        "weights_file": "attention_weights.safetensors",
        "submodules": {},
    }
    (tmp_path / "surgical_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    save_file({"completely_unknown.weight": torch.randn(4, 4)}, str(tmp_path / "attention_weights.safetensors"))

    with pytest.raises(RuntimeError, match="0 out of 1 tensors"):
        load_surgical_checkpoint(model, tmp_path, strict=False)
