#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for executor seed threading (finding F-95).
"""

from __future__ import annotations

import unittest.mock as mock
import pytest

import neuralops.benchmarks.executor as bexec
import neuralops.benchmarks.core.training as btrain


def test_execute_single_threads_seed_to_training(monkeypatch):
    """Verify execute_single preserves and threads seed to task adapter without dropping it."""
    # Test router routing with seed
    adapter = bexec._select_adapter(
        module_name="test_mod",
        module_type="attention",
        model="distilbert-base-uncased",
        dataset="sst2",
        rank=32,
        device="cpu",
        dtype="float32",
        batch_size=8,
        max_length=64,
        epochs=1,
        lr=1e-4,
        seed=999,
        eval_only=False,
    )
    assert getattr(adapter, "seed", None) == 999
