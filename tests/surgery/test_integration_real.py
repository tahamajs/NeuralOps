#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
End-to-end multi-stage real-model integration test on GPT-2.
"""

from __future__ import annotations

import copy
import time
import warnings

import torch

from neuralops.surgery.pipeline import SurgeryOp, SurgeryRecipe


def test_full_recipe_on_real_gpt2(gpt2_model, gpt2_tokenizer):
    """End-to-end integration test executing attention transplant and MLP shrinkage on real GPT-2."""
    model_copy = copy.deepcopy(gpt2_model)
    orig_params = sum(p.numel() for p in model_copy.parameters())

    recipe = SurgeryRecipe(
        ops=[
            SurgeryOp(name="replace_attention", params={"attention": "chunk_gla"}),
            SurgeryOp(name="shrink_mlp", params={"ratio": 0.85}),
        ],
        name="gpt2-real-surgery",
        base_model="gpt2",
    )

    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        surgeried_model, report = recipe.apply(model_copy, rollback_on_failure=True)
    dt = time.perf_counter() - t0

    new_params = sum(p.numel() for p in surgeried_model.parameters())
    reduction_pct = (orig_params - new_params) / orig_params * 100.0

    print(
        f"\n[BENCHMARK] Full Real GPT-2 Integration Run:\n"
        f"  Total Duration: {dt*1000:.2f}ms\n"
        f"  Parameters: {orig_params:,} -> {new_params:,} ({reduction_pct:.2f}% reduction)\n"
        f"  Applied Operations: {report.applied_ops}"
    )
    assert report.success
    assert new_params < orig_params
    assert reduction_pct >= 2.0

    # Test generation post surgery
    prompt = "NeuralOps enables"
    inputs = gpt2_tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = surgeried_model.generate(
            **inputs,
            max_new_tokens=6,
            do_sample=False,
            pad_token_id=gpt2_tokenizer.eos_token_id,
        )
    text = gpt2_tokenizer.decode(out[0], skip_special_tokens=True)
    print(f"  Generated Output: '{text}'")
    assert len(text) > len(prompt)
