#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
End-to-end pipeline and recipe validation tests on real GPT-2.
"""

from __future__ import annotations

import copy
import tempfile
import time
import warnings
from pathlib import Path

import pytest
import torch

from neuralops.surgery.pipeline import SurgeryOp, SurgeryRecipe
from neuralops.surgery.recipe import load_recipe, save_recipe


def test_pipeline_real_gpt2_user_story(gpt2_model, gpt2_tokenizer, short_text_inputs):
    """Prove the end-to-end user story: multi-stage recipe validation, application, and inference."""
    model_copy = copy.deepcopy(gpt2_model)
    params_before = sum(p.numel() for p in model_copy.parameters())

    recipe = SurgeryRecipe(
        ops=[
            SurgeryOp(name="replace_attention", params={"attention": "chunk_gla"}),
            SurgeryOp(name="shrink_mlp", params={"ratio": 0.90}),
        ],
        name="gpt2-fast-real",
        base_model="gpt2",
    )

    # 1. Pre-surgery validation
    t0_val = time.perf_counter()
    report = recipe.validate(model_copy)
    dt_val = time.perf_counter() - t0_val
    print(f"\n[BENCHMARK] Real GPT-2 Pre-Surgery Validation: {dt_val*1000:.2f}ms (is_safe={report.is_safe})")
    assert report.is_safe, f"Validation failed unexpectedly: {report.warnings}"

    # 2. Recipe application
    t0_app = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        transformed_model, apply_report = recipe.apply(model_copy, rollback_on_failure=True)
    dt_app = time.perf_counter() - t0_app

    print(
        f"[BENCHMARK] Real GPT-2 Recipe Execution: {dt_app*1000:.2f}ms\n"
        f"  Applied Operations: {apply_report.applied_ops}\n"
        f"  Success: {apply_report.success}"
    )
    assert apply_report.success
    assert "replace_attention" in apply_report.applied_ops
    assert "shrink_mlp" in apply_report.applied_ops

    # 3. Parameter reduction assertion
    params_after = sum(p.numel() for p in transformed_model.parameters())
    reduction_pct = (params_before - params_after) / params_before * 100.0
    print(
        f"  Parameters Before: {params_before:,}\n"
        f"  Parameters After: {params_after:,}\n"
        f"  Total Parameter Reduction: {reduction_pct:.2f}%"
    )
    assert params_after < params_before

    # 4. Post-surgery forward pass verification
    with torch.no_grad():
        out = transformed_model(**short_text_inputs)
        logits = out.logits
    assert torch.isfinite(logits).all()

    # 5. Post-surgery text generation verification
    inputs = gpt2_tokenizer("The neural operations", return_tensors="pt")
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        output_ids = transformed_model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
            pad_token_id=gpt2_tokenizer.eos_token_id,
        )
    assert output_ids.shape[1] == input_len + 8
    text = gpt2_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"  Post-Surgery Generation: '{text}'")
    assert len(text) > 0


def test_pipeline_recipe_yaml_roundtrip():
    """Verify recipe YAML serialization roundtrip fidelity."""
    recipe = SurgeryRecipe(
        ops=[
            SurgeryOp(name="replace_attention", params={"attention": "chunk_gla"}),
            SurgeryOp(name="shrink_mlp", params={"ratio": 0.85}),
        ],
        name="roundtrip-spec",
        base_model="gpt2",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir) / "test_recipe.yaml"
        save_recipe(recipe, p)
        assert p.is_file()

        loaded = load_recipe(p)
        assert loaded.name == recipe.name
        assert loaded.base_model == recipe.base_model
        assert len(loaded.ops) == 2
        assert loaded.ops[0].name == "replace_attention"
        assert loaded.ops[1].params["ratio"] == 0.85


def test_pipeline_composition_conflict_rejection(gpt2_model):
    """Verify validation gates reject conflicting consecutive operations targeting the same subsystem."""
    model_copy = copy.deepcopy(gpt2_model)
    conflicting_recipe = SurgeryRecipe(
        ops=[
            SurgeryOp(name="shrink_mlp", params={"ratio": 0.9}),
            SurgeryOp(name="shrink_mlp", params={"ratio": 0.8}),
        ]
    )

    report = conflicting_recipe.validate(model_copy)
    assert not report.is_safe
    compat_check = next(c for c in report.checks if c.name == "recipe_compat")
    assert not compat_check.passed
    assert "Multiple operations modify the same category" in compat_check.details
