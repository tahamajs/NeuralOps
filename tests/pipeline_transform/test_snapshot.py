#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for Sub-Agent B2: Snapshot Agent (Selective State Capture & Bit-Identical Rollback).
"""

from __future__ import annotations

import hashlib

import torch
import torch.nn as nn

from neuralops.pipeline_transform.snapshot import capture, restore


def _hash_state_dict(sd: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for k in sorted(sd.keys()):
        h.update(k.encode("utf-8"))
        h.update(sd[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()


class MultiLayerDummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # 10 linear layers of size 128x128
        self.layers = nn.ModuleList([nn.Linear(128, 128) for _ in range(10)])
        self.config = type("Config", (), {"name": "multi-layer-dummy", "lr": 1e-4})()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def test_b2_capture_and_restore_bit_identical():
    """Verify capture + restore leaves state_dict bit-identical."""
    model = MultiLayerDummyModel()
    hash_before = _hash_state_dict(model.state_dict())

    # Capture snapshot
    snap = capture(model)

    # Intentionally mutate weights
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.234)

    hash_mutated = _hash_state_dict(model.state_dict())
    assert hash_mutated != hash_before

    # Restore
    restore(model, snap)

    hash_after = _hash_state_dict(model.state_dict())
    assert hash_after == hash_before


def test_b2_selective_capture_memory_proportional():
    """Verify capture on a small module only copies that module, not the full model."""
    model = MultiLayerDummyModel()

    # Total model parameter bytes
    total_model_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    # Selectively capture ONLY layer 0
    snap_selective = capture(model, state_paths=["layers.0"])
    selective_bytes = snap_selective.size_bytes()

    # 1 layer out of 10 should be ~10% of total bytes
    assert selective_bytes > 0
    assert selective_bytes < (total_model_bytes * 0.20)
    assert len(snap_selective.params) == 2  # weight + bias of layer 0 only


def test_b2_structural_mutation_rollback():
    """Verify rollback after replacing a submodule with a different architecture works."""
    model = MultiLayerDummyModel()
    hash_before = _hash_state_dict(model.state_dict())

    # Snapshot with targeted path
    snap = capture(model, state_paths=["layers.0"])

    # Structurally replace layer 0 with Sequential
    replacement = nn.Sequential(
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 128),
    )
    model.layers[0] = replacement

    # Verify structural change occurred
    assert isinstance(model.layers[0], nn.Sequential)

    # Roll back
    restore(model, snap)

    # Verify original module class and bit-identical state are restored
    assert isinstance(model.layers[0], nn.Linear)
    hash_after = _hash_state_dict(model.state_dict())
    assert hash_after == hash_before


def test_b2_config_attribute_rollback():
    """Verify config mutations are reverted during rollback."""
    model = MultiLayerDummyModel()
    assert model.config.lr == 1e-4

    snap = capture(model)
    model.config.lr = 0.05

    restore(model, snap)
    assert model.config.lr == 1e-4


def test_b2_spill_to_disk_safetensors_and_restore():
    """Verify disk-spilled SafeTensors snapshot frees RAM and restores bit-identical state."""
    model = MultiLayerDummyModel()
    hash_before = _hash_state_dict(model.state_dict())

    # Capture snapshot with spill_to_disk=True
    snap = capture(model, spill_to_disk=True)

    assert snap.disk_path is not None
    assert len(snap.params) == 0  # In-memory params released
    assert snap.size_bytes() > 0

    # Intentionally mutate weights
    with torch.no_grad():
        for p in model.parameters():
            p.add_(9.876)

    hash_mutated = _hash_state_dict(model.state_dict())
    assert hash_mutated != hash_before

    # Restore from disk SafeTensors
    restore(model, snap)

    hash_after = _hash_state_dict(model.state_dict())
    assert hash_after == hash_before

    # Cleanup disk resources
    snap.cleanup()
    assert snap.disk_path is None
