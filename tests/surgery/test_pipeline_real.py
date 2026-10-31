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
import os
import tempfile
import warnings
from pathlib import Path

import torch

from neuralops.surgery.pipeline import SurgeryOp, SurgeryRecipe
from neuralops.surgery.recipe import load_recipe, save_recipe


def test_full_recipe_on_gpt2(gpt2_model, gpt2_tokenizer, gpt2_inputs):
    """End-to-end recipe: replace_attention + shrink_mlp on real GPT-2."""
    model = copy.deepcopy(gpt2_model)
    before_params = sum(p.numel() for p in model.parameters())

    recipe = SurgeryRecipe(
        ops=[
            SurgeryOp(name="replace_attention", params={"attention": "chunk_gla"}),
            SurgeryOp(name="shrink_mlp", params={"ratio": 0.9}),
        ],
        name="gpt2_compact",
        base_model="gpt2",
    )

    # 1. Validation
    report = recipe.validate(model)
    assert report.is_safe is True

    # 2. Application
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        new_model, apply_report = recipe.apply(model, rollback_on_failure=True)
    assert apply_report.success is True
    assert "replace_attention" in apply_report.applied_ops
    assert "shrink_mlp" in apply_report.applied_ops

    # 3. Parameter reduction verification
    after_params = sum(p.numel() for p in new_model.parameters())
    assert after_params < before_params
    reduction = 100.0 * (1.0 - after_params / before_params)

    # 4. Forward pass
    with torch.no_grad():
        out = new_model(**gpt2_inputs)
        logits = out.logits
    assert torch.isfinite(logits).all()

    # 5. Generate 5 tokens
    input_ids = gpt2_inputs["input_ids"]
    with torch.no_grad():
        gen = new_model.generate(
            input_ids=input_ids,
            max_new_tokens=5,
            do_sample=False,
            pad_token_id=gpt2_tokenizer.eos_token_id,
        )
    assert gen.shape[1] == input_ids.shape[1] + 5
    text = gpt2_tokenizer.decode(gen[0], skip_special_tokens=True)
    assert len(text) > 0
    print(f"\n[BENCHMARK] Recipe applied successfully:")
    print(f"  Params: {before_params:,} -> {after_params:,} ({reduction:.2f}% reduction)")
    print(f"  Generation: '{text}'")


def test_recipe_yaml_roundtrip(gpt2_model):
    """Verify recipe serializes to YAML and applies accurately after reloading."""
    recipe = SurgeryRecipe(
        ops=[
            SurgeryOp(name="replace_attention", params={"attention": "chunk_gla"}),
            SurgeryOp(name="shrink_mlp", params={"ratio": 0.9}),
        ],
        name="roundtrip_recipe",
        base_model="gpt2",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "test_recipe.yaml"
        save_recipe(recipe, tmp_path)
        assert tmp_path.is_file()

        loaded = load_recipe(tmp_path)
        assert loaded.name == recipe.name
        assert loaded.base_model == recipe.base_model
        assert len(loaded.ops) == len(recipe.ops)
        assert loaded.ops[0].name == recipe.ops[0].name
        assert loaded.ops[0].params == recipe.ops[0].params
        assert loaded.ops[1].name == recipe.ops[1].name
        assert loaded.ops[1].params == recipe.ops[1].params

        model = copy.deepcopy(gpt2_model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            new_model, report = loaded.apply(model, rollback_on_failure=True)
        assert report.success is True
        assert "replace_attention" in report.applied_ops
        assert "shrink_mlp" in report.applied_ops


def test_recipe_validation_rejects_conflicts(gpt2_model):
    """Verify conflicting operations are safely rejected before applying."""
    recipe = SurgeryRecipe(
        ops=[
            SurgeryOp(name="replace_attention", params={"attention": "chunk_gla"}),
            SurgeryOp(name="replace_attention", params={"attention": "cosformer"}),
        ],
        name="conflicting_recipe",
        base_model="gpt2",
    )

    report = recipe.validate(gpt2_model)
    assert report.is_safe is False
    compat_check = next(c for c in report.checks if c.name == "recipe_compat")
    assert not compat_check.passed
    assert "Multiple operations modify the same category" in compat_check.details


def test_recipe_rollback_on_failure(gpt2_model):
    """Verify model state is restored when a recipe operation fails."""
    model = copy.deepcopy(gpt2_model)
    before_params = sum(p.numel() for p in model.parameters())
    orig_state = {name: p.clone() for name, p in model.named_parameters()}

    recipe = SurgeryRecipe(
        ops=[
            SurgeryOp(name="shrink_mlp", params={"ratio": 0.9}),
            SurgeryOp(name="quantize_int4", params={"method": "awq"}),
        ],
        name="failing_recipe",
        base_model="gpt2",
    )

    res_model, report = recipe.apply(model, rollback_on_failure=True)
    assert report.success is False
    assert "shrink_mlp" in report.applied_ops
    assert "quantize_int4" not in report.applied_ops
    assert report.error_message is not None

    after_params = sum(p.numel() for p in res_model.parameters())
    assert after_params == before_params

    for name, p in res_model.named_parameters():
        assert torch.equal(p, orig_state[name])
