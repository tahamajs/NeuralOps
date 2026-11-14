#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for ensuring save_surgical_delta preserves pre-existing user files."""

from __future__ import annotations

from pathlib import Path
import torch
import torch.nn as nn

from neuralops.models.surgical_io import save_surgical_delta


def test_save_surgical_delta_preserves_pre_existing_files(tmp_path: Path) -> None:
    # Create pre-existing heavy weight files that a user might have
    heavy_files = {
        "model.safetensors": b"pre-existing user model weights",
        "pytorch_model.bin": b"pre-existing binary weights",
        "surgical_model_state.pt": b"pre-existing pt state",
        "optimizer.pt": b"pre-existing optimizer state",
    }
    for filename, content in heavy_files.items():
        (tmp_path / filename).write_bytes(content)

    # Create a simple toy model
    model = nn.Sequential(nn.Linear(16, 16), nn.Linear(16, 16))
    for p in model.parameters():
        p.requires_grad = True

    # Save surgical delta with remove_full_weights=True
    save_surgical_delta(
        model,
        tmp_path,
        weights_filename="attention_weights.safetensors",
        remove_full_weights=True,
    )

    # Verify that pre-existing files were preserved and NOT unlinked
    for filename, content in heavy_files.items():
        file_path = tmp_path / filename
        assert file_path.exists(), f"Pre-existing file '{filename}' was deleted!"
        assert file_path.read_bytes() == content, f"Pre-existing file '{filename}' content was altered!"

    # Verify surgical delta and manifest were created
    assert (tmp_path / "attention_weights.safetensors").exists()
    assert (tmp_path / "surgical_manifest.json").exists()
