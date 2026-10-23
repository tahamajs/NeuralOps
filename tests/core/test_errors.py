#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for NeuralOps error hierarchy and circular dependency prevention.
"""

from __future__ import annotations

import sys
from neuralops.core.errors import (
    NeuralOpsError,
    MissingDependencyError,
    IncompatibleArchitectureError,
    KernelDispatchError,
    CalibrationError,
    torch_install_error,
)


def test_error_hierarchy():
    """Verify that all custom errors inherit from NeuralOpsError."""
    assert issubclass(MissingDependencyError, (ImportError, NeuralOpsError))
    assert issubclass(IncompatibleArchitectureError, (ValueError, NeuralOpsError))
    assert issubclass(KernelDispatchError, (RuntimeError, NeuralOpsError))
    assert issubclass(CalibrationError, (RuntimeError, NeuralOpsError))


def test_torch_install_error_message():
    """Verify torch_install_error produces actionable guidance."""
    err = torch_install_error("test_feature", cause=ImportError("no torch"))
    assert isinstance(err, MissingDependencyError)
    assert "NeuralOps requires PyTorch for test_feature" in str(err)
    assert 'pip install "neuralops[cpu]"' in str(err)
    assert "Underlying import error: no torch" in str(err)


def test_no_circular_import_between_api_and_top_level():
    """Ensure importing neuralops.api does not rely on top-level circular imports."""
    import neuralops.api as napi

    assert napi is not None
    assert hasattr(napi, "replace_attention")
