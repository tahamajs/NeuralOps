#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Unit tests for CheckpointManager utility."""

from __future__ import annotations

import time
from pathlib import Path

from neuralops.utils.checkpoint import CheckpointManager


def test_checkpoint_manager_lifecycle(tmp_path):
    """Verify save, load, list, get_latest, and delete lifecycle."""
    ckpt_dir = tmp_path / "ckpts"
    mgr = CheckpointManager(checkpoint_dir=str(ckpt_dir))

    # Initial state
    assert mgr.list_checkpoints() == []
    assert mgr.get_latest_checkpoint() is None

    # Save first checkpoint
    results_1 = [{"step": 100, "loss": 0.5}]
    p1 = mgr.save_checkpoint(
        experiment_id=1,
        results=results_1,
        checkpoint_name="ckpt_01.json",
        metadata={"model": "gpt2"},
    )
    assert p1.exists()

    # Load first checkpoint
    exp_id, res, meta = mgr.load_checkpoint("ckpt_01.json")
    assert exp_id == 1
    assert res == results_1
    assert meta == {"model": "gpt2"}

    # Load non-existent checkpoint returns defaults
    exp_id_def, res_def, meta_def = mgr.load_checkpoint("non_existent.json")
    assert exp_id_def == 0
    assert res_def == []
    assert meta_def is None

    time.sleep(0.01)

    # Save second checkpoint
    results_2 = [{"step": 200, "loss": 0.3}]
    p2 = mgr.save_checkpoint(
        experiment_id=2,
        results=results_2,
        checkpoint_name="ckpt_02.json",
    )
    assert p2.exists()

    all_ckpts = mgr.list_checkpoints()
    assert len(all_ckpts) == 2

    latest = mgr.get_latest_checkpoint()
    assert latest == p2

    # Delete first checkpoint
    mgr.delete_checkpoint("ckpt_01.json")
    assert not p1.exists()
    assert len(mgr.list_checkpoints()) == 1
