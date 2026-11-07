#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests verifying canonical transform execution stack and recipe disambiguation (F-56).
"""

from __future__ import annotations

import pytest

import neuralops.recipe.schema as recipe_schema
import neuralops.surgery.recipe as surgery_recipe
from neuralops.pipeline_transform.plan import ExecutionPlan as CanonicalExecutionPlan


def test_both_stacks_agree_on_types():
    """Verify ExecutionPlan from neuralops.executor is identical to canonical pipeline_transform.plan."""
    from neuralops.executor.plan import ExecutionPlan as ExecutorPlan

    assert ExecutorPlan is CanonicalExecutionPlan


def test_deprecated_stack_warns():
    """Verify importing from neuralops.executor emits a DeprecationWarning."""
    import importlib
    import warnings
    import neuralops.executor

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always", DeprecationWarning)
        importlib.reload(neuralops.executor)

    assert any(
        issubclass(w.category, DeprecationWarning) and "neuralops.executor is deprecated" in str(w.message)
        for w in recorded
    ), "Expected DeprecationWarning when importing neuralops.executor"


def test_recipe_concepts_distinct():
    """Verify neuralops.recipe and neuralops.surgery.recipe have distinct explicit docstrings."""
    recipe_doc = recipe_schema.__doc__ or ""
    surgery_doc = surgery_recipe.__doc__ or ""

    assert "NRL" in recipe_doc or "Recipe Language" in recipe_doc
    assert "Surgery" in surgery_doc
    assert "Distinct from" in recipe_doc or "Scope Distinction" in surgery_doc
