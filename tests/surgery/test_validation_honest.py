#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for honest validation estimates in validate_before_surgery.
Ensures estimated_vram_reduction_mb is None, estimated_latency_change_pct is None,
quality_passed is None, and requires_empirical_validation is True.
"""

from neuralops.surgery.pipeline import SurgeryOp, SurgeryRecipe
from neuralops.surgery.registry import SURGERY_REGISTRY, register_surgery
from neuralops.surgery.validation import validate_before_surgery


def test_validation_estimates_are_honest():
    """Verify that validation does not report fake estimates without empirical measurement."""

    @register_surgery(name="honest_op", category="attention", compat=["gpt2", "llama"])
    def honest_op(model):
        return model

    recipe = SurgeryRecipe([SurgeryOp(name="honest_op", params={})])
    report = validate_before_surgery(None, recipe)

    assert report.is_safe is True
    assert report.estimated_vram_reduction_mb is None
    assert report.estimated_latency_change_pct is None
    assert report.requires_empirical_validation is True

    quality_check = next(c for c in report.checks if c.name == "quality_probe")
    assert quality_check.passed is None

    del SURGERY_REGISTRY["honest_op"]
