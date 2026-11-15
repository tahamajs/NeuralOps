#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Regression tests for neuralops.core.inspector."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

from neuralops.core import inspector


def test_inspect_architecture_known_module() -> None:
    """Verify inspect_architecture returns valid ArchitectureSpec for known module."""
    spec = inspector.inspect_architecture("full_attention")
    assert spec is not None
    assert spec.name.lower() == "full_attention"
    assert spec.family
    assert spec.time_complexity
    assert len(spec.constructor_params) > 0


def test_inspect_architecture_unknown_module_returns_default_spec() -> None:
    """Verify inspect_architecture returns a default ArchitectureSpec for unknown modules without crashing."""
    spec = inspector.inspect_architecture("completely_unknown_custom_layer_123")
    assert spec is not None
    assert spec.name == "completely_unknown_custom_layer_123"
    assert spec.family == "custom-unknown"
    assert spec.time_complexity == "O(T·d)"
    assert len(spec.constructor_params) > 0
    assert isinstance(spec.to_dict(), dict)
