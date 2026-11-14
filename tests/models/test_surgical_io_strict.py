#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for load_surgical_checkpoint critical parameter validation."""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

from neuralops.models.surgical_io import load_surgical_checkpoint


class ModelWithClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(16, 16)
        self.classifier = nn.Linear(16, 2)


class ModelWithLMHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(50, 16)
        self.lm_head = nn.Linear(16, 50, bias=False)


def test_load_surgical_checkpoint_missing_classifier_raises(tmp_path: Path) -> None:
    # Save checkpoint with encoder but missing classifier.weight
    state = {
        "encoder.weight": torch.randn(16, 16),
        "encoder.bias": torch.zeros(16),
    }
    save_file(state, str(tmp_path / "model.safetensors"))
    manifest = {
        "schema_version": "neuralops.surgery/v1",
        "submodules": {},
    }
    (tmp_path / "surgical_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    model = ModelWithClassifier()

    with pytest.raises(RuntimeError, match="Missing critical parameters during checkpoint load"):
        load_surgical_checkpoint(model, tmp_path, strict=False)


def test_load_surgical_checkpoint_missing_lm_head_raises(tmp_path: Path) -> None:
    # Save checkpoint missing lm_head.weight
    state = {
        "embed.weight": torch.randn(50, 16),
    }
    save_file(state, str(tmp_path / "model.safetensors"))
    manifest = {
        "schema_version": "neuralops.surgery/v1",
        "submodules": {},
    }
    (tmp_path / "surgical_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    model = ModelWithLMHead()

    with pytest.raises(RuntimeError, match="Missing critical parameters during checkpoint load"):
        load_surgical_checkpoint(model, tmp_path, strict=False)


def test_load_surgical_checkpoint_with_all_critical_params_succeeds(tmp_path: Path) -> None:
    # Save checkpoint with classifier included
    state = {
        "encoder.weight": torch.randn(16, 16),
        "encoder.bias": torch.zeros(16),
        "classifier.weight": torch.randn(2, 16),
        "classifier.bias": torch.zeros(2),
    }
    save_file(state, str(tmp_path / "model.safetensors"))
    manifest = {
        "schema_version": "neuralops.surgery/v1",
        "submodules": {},
    }
    (tmp_path / "surgical_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    model = ModelWithClassifier()
    loaded_model, loaded_manifest = load_surgical_checkpoint(model, tmp_path, strict=False)
    assert loaded_model is model
