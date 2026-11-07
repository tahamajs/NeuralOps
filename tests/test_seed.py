#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for canonical set_seed helper (finding F-94).
"""

from __future__ import annotations

import random
import numpy as np
import pytest
import torch

from neuralops.core.utils.general import set_seed
from neuralops.utils.seed import set_all_seeds


def test_set_seed_is_deterministic():
    """Verify seeding twice with the same value produces identical random streams."""
    set_seed(12345, deterministic=True)
    r1 = [random.random() for _ in range(5)]
    n1 = np.random.rand(5).tolist()
    t1 = torch.randn(5).tolist()

    set_seed(12345, deterministic=True)
    r2 = [random.random() for _ in range(5)]
    n2 = np.random.rand(5).tolist()
    t2 = torch.randn(5).tolist()

    assert r1 == r2
    assert n1 == n2
    assert t1 == t2


def test_set_all_seeds_is_same_as_set_seed():
    """Verify utils.seed.set_all_seeds is an identical re-export of core set_seed."""
    assert set_all_seeds is set_seed


def test_set_seed_guards_accelerator_when_unavailable(monkeypatch):
    """Verify set_seed executes cleanly without errors when GPU accelerator is unavailable."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    # Should not raise any exceptions even if accelerator is absent
    set_seed(42, deterministic=True)
