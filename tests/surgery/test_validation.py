#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit tests for surgery validation gates.
"""

from neuralops.surgery.pipeline import SurgeryOp, SurgeryRecipe
from neuralops.surgery.registry import SURGERY_REGISTRY, register_surgery
from neuralops.surgery.validation import validate_before_surgery


def test_validation_checks_pass():
    @register_surgery(name="val_op", category="attention", compat=["gpt2", "llama"])
    def val_op(model):
        return model

    recipe = SurgeryRecipe([SurgeryOp(name="val_op", params={})])
    report = validate_before_surgery(None, recipe)

    assert report.is_safe
    assert len(report.checks) == 5
    assert all(c.passed for c in report.checks)

    del SURGERY_REGISTRY["val_op"]


def test_validation_conflicting_surgeries_fail():
    @register_surgery(name="val_mlp1", category="mlp")
    def val_mlp1(model):
        return model

    @register_surgery(name="val_mlp2", category="mlp")
    def val_mlp2(model):
        return model

    conflicting_recipe = SurgeryRecipe([
        SurgeryOp(name="val_mlp1", params={}),
        SurgeryOp(name="val_mlp2", params={}),
    ])

    report = validate_before_surgery(None, conflicting_recipe)
    assert not report.is_safe
    compat_check = next(c for c in report.checks if c.name == "recipe_compat")
    assert not compat_check.passed

    del SURGERY_REGISTRY["val_mlp1"]
    del SURGERY_REGISTRY["val_mlp2"]
