#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for install trap prevention and torch-free behavior (finding F-96).
"""

from __future__ import annotations

import builtins
import sys
import pytest

import neuralops
from neuralops.core.exceptions import MissingDependencyError, torch_install_error


def test_pipeline_not_torch_free():
    """Verify that 'pipeline' is not included in _TORCH_FREE_SYMBOLS."""
    import neuralops
    torch_free = getattr(neuralops, "_TORCH_FREE_SYMBOLS", set())
    assert "pipeline" not in torch_free, "'pipeline' must not be advertised as torch-free"


def test_torch_error_points_to_core_extra():
    """Verify torch_install_error message instructs the user to install neuralops[core]."""
    err = torch_install_error("neuralops.pipeline")
    assert isinstance(err, (ImportError, MissingDependencyError))
    err_msg = str(err)
    assert 'pip install "neuralops[core]"' in err_msg or "neuralops[core]" in err_msg


def test_require_torch_fails_loudly_when_torch_missing(monkeypatch):
    """Verify _require_torch raises an error pointing at neuralops[core] when torch is unavailable."""
    monkeypatch.setattr(neuralops, "_TORCH_AVAILABLE", False)
    with pytest.raises((ImportError, MissingDependencyError)) as exc_info:
        neuralops._require_torch("test_feature")
    assert "neuralops[core]" in str(exc_info.value)
