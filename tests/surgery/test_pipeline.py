#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit tests for surgery pipeline and recipe execution.
"""

from neuralops.surgery.pipeline import SurgeryOp, SurgeryRecipe
from neuralops.surgery.registry import SURGERY_REGISTRY, register_surgery


def test_surgery_op_serialization():
    op = SurgeryOp(name="replace_attention", params={"attention": "chunk_gla"})
    d = op.to_dict()
    assert d["name"] == "replace_attention"
    assert d["params"]["attention"] == "chunk_gla"

    op2 = SurgeryOp.from_dict(d)
    assert op2.name == op.name
    assert op2.params == op.params


def test_recipe_apply_and_rollback():
    try:
        @register_surgery(name="pipe_op1", category="attention")
        def pipe_op1(model, val=1):
            model["count"] += val
            return model

        @register_surgery(name="pipe_op_fail", category="custom_fail")
        def pipe_op_fail(model):
            raise ValueError("Simulated failure")

        recipe = SurgeryRecipe([
            SurgeryOp(name="pipe_op1", params={"val": 5}),
        ])

        model = {"count": 10}
        new_model, report = recipe.apply(model)
        assert report.success
        assert new_model["count"] == 15
        assert "pipe_op1" in report.applied_ops

        # Test failure with rollback
        fail_recipe = SurgeryRecipe([
            SurgeryOp(name="pipe_op1", params={"val": 5}),
            SurgeryOp(name="pipe_op_fail", params={}),
        ])

        orig_model = {"count": 100}
        res_model, fail_report = fail_recipe.apply(orig_model, rollback_on_failure=True)
        assert not fail_report.success
        assert res_model["count"] == 100
    finally:
        SURGERY_REGISTRY.pop("pipe_op1", None)
        SURGERY_REGISTRY.pop("pipe_op_fail", None)
