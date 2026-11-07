#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests verifying evaluation package facade and export consistency (F-54).
"""

from __future__ import annotations

from pathlib import Path

import pytest

import neuralops.evaluation as ev


def test_no_none_exports():
    """Verify that every symbol listed in __all__ resolves to a real object (never None)."""
    for name in ev.__all__:
        val = getattr(ev, name)
        assert val is not None, f"Export {name} in neuralops.evaluation resolved to None"


def test_all_modules_exported():
    """Verify that every .py file in neuralops/evaluation/ (except __init__.py) is exported."""
    eval_dir = Path(ev.__file__).parent
    eval_py_files = {
        f.stem
        for f in eval_dir.glob("*.py")
        if f.stem not in ("__init__", "__main__")
    }

    assert eval_py_files.issubset(set(ev._MODULE_EXPORTS.keys())), (
        f"Missing modules in _MODULE_EXPORTS: {eval_py_files - set(ev._MODULE_EXPORTS.keys())}"
    )


def test_missing_attribute_raises_attribute_error():
    """Verify that accessing a missing attribute raises AttributeError (never None)."""
    with pytest.raises(AttributeError):
        _ = ev.NonExistentEvaluationSymbolXyz123

    with pytest.raises(AttributeError):
        _ = ev.RobustnessReport
