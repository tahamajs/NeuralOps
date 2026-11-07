#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for public API seed parameter threading (finding F-95).
"""

from __future__ import annotations

import unittest.mock as mock
import pytest
import torch
import torch.nn as nn

import neuralops.api as napi
import neuralops.benchmarks.core.training as btrain
from neuralops.core.utils import set_seed
from examples._shared.mock_models import TinyClassifier


def test_api_train_threads_seed_to_finetune(monkeypatch):
    """Verify api.train(..., seed=123) threads seed=123 to finetune_classification."""
    calls = []

    def mock_finetune(model, train_loader, **kwargs):
        calls.append(kwargs)
        return {"status": "ok", "loss": 0.5}

    monkeypatch.setattr(btrain, "finetune_classification", mock_finetune)

    model = nn.Linear(10, 2)
    dataset = [torch.randn(4, 10) for _ in range(2)]

    res = napi.train(model=model, dataset=dataset, seed=123)
    assert len(calls) == 1
    assert calls[0].get("seed") == 123


def test_api_train_default_seed_is_42(monkeypatch):
    """Verify api.train defaults to seed 42 when omitted."""
    calls = []

    def mock_finetune(model, train_loader, **kwargs):
        calls.append(kwargs)
        return {"status": "ok"}

    monkeypatch.setattr(btrain, "finetune_classification", mock_finetune)

    model = nn.Linear(10, 2)
    dataset = [torch.randn(4, 10) for _ in range(2)]

    napi.train(model=model, dataset=dataset)
    assert len(calls) == 1
    assert calls[0].get("seed") == 42


def test_finetune_classification_calls_set_seed(monkeypatch):
    """Verify finetune_classification calls set_seed with the provided seed."""
    seeded = []

    def mock_set_seed(s, **kwargs):
        seeded.append(s)

    import neuralops.core.utils.general as core_gen
    monkeypatch.setattr(core_gen, "set_seed", mock_set_seed)

    model = nn.Linear(4, 2)
    loader = []

    btrain.finetune_classification(model=model, train_loader=loader, epochs=0, seed=7)
    assert 7 in seeded


def test_two_runs_same_seed_same_init():
    """Verify two runs initialized with the same seed give identical model weights."""
    set_seed(42)
    m1 = TinyClassifier(dim=32, num_labels=2)

    set_seed(42)
    m2 = TinyClassifier(dim=32, num_labels=2)

    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert torch.equal(p1, p2)
