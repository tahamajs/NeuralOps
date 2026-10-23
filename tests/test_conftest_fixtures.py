#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests verifying test infrastructure fixtures, seed resets, and marker registration."""
from __future__ import annotations

import random
import pytest
import torch


def test_tiny_gpt2_fixture(tiny_gpt2):
    """Verify tiny_gpt2 fixture produces a valid module with forward capability."""
    assert tiny_gpt2 is not None
    assert hasattr(tiny_gpt2, "config")
    dummy_input = torch.randint(0, 100, (1, 8))
    out = tiny_gpt2(dummy_input)
    assert out is not None
    assert hasattr(out, "logits")
    assert out.logits.shape[0] == 1
    assert out.logits.shape[1] == 8


def test_reset_torch_seed_determinism():
    """Verify that random seeds are reset for determinism across tests."""
    val1 = torch.randn(4)
    torch.manual_seed(42)
    val2 = torch.randn(4)
    assert torch.allclose(val1, val2)


@pytest.mark.smoke
def test_smoke_marker():
    """Verify smoke marker is registered and does not emit warnings."""
    assert True


@pytest.mark.slow
def test_slow_marker():
    """Verify slow marker is registered and does not emit warnings."""
    assert True


@pytest.mark.integration
def test_integration_marker():
    """Verify integration marker is registered and does not emit warnings."""
    assert True
