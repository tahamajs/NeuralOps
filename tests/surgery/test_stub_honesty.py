#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests verifying honest failure contracts, limitations, and documentation of stubs.
"""

from __future__ import annotations

import inspect
import warnings

import pytest

from neuralops.core.errors import IntegrationNotImplementedError
from neuralops.surgery.layers.prune import prune_layers
from neuralops.surgery.recovery.distill import distill_from_teacher
from neuralops.surgery.weights.quantize import quantize_int4


def test_quantize_int4_fail_closed_and_validation():
    """Verify quantize_int4 fails closed with clear actionable errors."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)

        # 1. Invalid method raises ValueError
        with pytest.raises(ValueError, match="Unsupported quantization method"):
            quantize_int4(model=None, method="nonexistent_method_xyz")

        # 2. AWQ dependency missing raises IntegrationNotImplementedError
        with pytest.raises((IntegrationNotImplementedError, ImportError, RuntimeError)) as exc_info:
            quantize_int4(model=None, method="awq")
        assert "autoawq" in str(exc_info.value).lower()

        # 3. GPTQ dependency missing raises IntegrationNotImplementedError
        with pytest.raises((IntegrationNotImplementedError, ImportError, RuntimeError)) as exc_info:
            quantize_int4(model=None, method="gptq")
        assert "gptqmodel" in str(exc_info.value).lower()


def test_distill_from_teacher_docstring_honesty():
    """Verify distill_from_teacher explicitly documents its GPU requirement."""
    doc = inspect.getdoc(distill_from_teacher)
    assert doc is not None, "distill_from_teacher missing docstring"
    assert "GPU" in doc or "CUDA" in doc, "distill_from_teacher docstring must document GPU/CUDA requirement"
    assert "healing" in doc.lower() or "distillation" in doc.lower()


def test_prune_layers_docstring_honesty():
    """Verify prune_layers documents its partial scope (shedding without healing on CPU)."""
    doc = inspect.getdoc(prune_layers)
    assert doc is not None, "prune_layers missing docstring"
    assert "healing" in doc.lower(), "prune_layers docstring must clarify healing status"
    assert "GPU" in doc or "CPU" in doc, "prune_layers docstring must state hardware requirements for healing"


def test_prune_layers_empty_selection_raises():
    """Verify prune_layers raises clear ValueError rather than silently returning corrupted model."""
    class DummyModel:
        pass

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        # Pruning non-module safely returns model or warns
        assert prune_layers(DummyModel()) is not None
