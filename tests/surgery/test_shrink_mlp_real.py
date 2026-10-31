#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Real-model validation tests for shrink_mlp surgery on GPT-2 and DistilBERT.
"""

from __future__ import annotations

import copy
import time
import warnings

import pytest
import torch

from neuralops.surgery.mlp.shrink import shrink_mlp


def test_shrink_mlp_gpt2_parameter_reduction(gpt2_model, short_text_inputs):
    """Prove shrink_mlp reduces parameter count on real GPT-2 while preserving representation."""
    model_copy = copy.deepcopy(gpt2_model)
    params_before = sum(p.numel() for p in model_copy.parameters())

    # Baseline forward logits
    with torch.no_grad():
        base_logits = model_copy(**short_text_inputs).logits

    # Apply shrink_mlp
    t0_surg = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        shrunk_model = shrink_mlp(model_copy, ratio=0.8)
    dt_surg = time.perf_counter() - t0_surg

    params_after = sum(p.numel() for p in shrunk_model.parameters())
    reduction_pct = (params_before - params_after) / params_before * 100.0

    print(
        f"\n[BENCHMARK] GPT-2 shrink_mlp(ratio=0.8):\n"
        f"  Surgery Duration: {dt_surg*1000:.2f}ms\n"
        f"  Parameters Before: {params_before:,}\n"
        f"  Parameters After: {params_after:,}\n"
        f"  Actual Reduction: {reduction_pct:.2f}%"
    )

    # 1. Parameter assertions
    assert params_after < params_before, "Parameters did not decrease after MLP shrinkage"
    assert 2.0 <= reduction_pct <= 40.0, f"Reduction {reduction_pct:.2f}% outside expected 2%-40% range"

    # 2. Forward pass validity
    with torch.no_grad():
        shrunk_logits = shrunk_model(**short_text_inputs).logits

    assert shrunk_logits.shape == base_logits.shape
    assert torch.isfinite(shrunk_logits).all()

    # 3. Representation preservation (cosine similarity)
    cos_sim = torch.nn.functional.cosine_similarity(
        base_logits.flatten(), shrunk_logits.flatten(), dim=0
    ).item()
    print(f"  Logits Cosine Similarity: {cos_sim:.4f}")
    assert cos_sim > 0.5, f"Cosine similarity {cos_sim:.4f} too low (expected > 0.5)"


def test_shrink_mlp_gpt2_perplexity(gpt2_model, gpt2_tokenizer):
    """Test perplexity degradation of real GPT-2 before and after gentle MLP shrinkage."""
    model_copy = copy.deepcopy(gpt2_model)
    sample_text = (
        "The quick brown fox jumps over the lazy dog. "
        "The dog was very lazy and did not move at all."
    )
    inputs = gpt2_tokenizer(sample_text, return_tensors="pt")

    # Baseline perplexity
    with torch.no_grad():
        loss_before = model_copy(inputs["input_ids"], labels=inputs["input_ids"]).loss
        ppl_before = torch.exp(loss_before).item()

    # Gentle shrinkage (ratio=0.90)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        shrunk = shrink_mlp(model_copy, ratio=0.90)

    # Post-shrink perplexity
    with torch.no_grad():
        loss_after = shrunk(inputs["input_ids"], labels=inputs["input_ids"]).loss
        ppl_after = torch.exp(loss_after).item()

    factor = ppl_after / ppl_before
    print(
        f"\n[BENCHMARK] GPT-2 Perplexity Retention (ratio=0.90):\n"
        f"  Perplexity Before: {ppl_before:.2f}\n"
        f"  Perplexity After: {ppl_after:.2f}\n"
        f"  Perplexity Change Factor: {factor:.2f}x"
    )
    assert ppl_after < 1.20 * ppl_before, f"Perplexity degradation ({factor:.2f}x) exceeded 20% budget"


def test_shrink_mlp_distilbert_encoder(distilbert_model, distilbert_inputs):
    """Test parameter reduction and forward pass on real DistilBERT 66M."""
    model_copy = copy.deepcopy(distilbert_model)
    params_before = sum(p.numel() for p in model_copy.parameters())

    with torch.no_grad():
        base_hidden = model_copy(**distilbert_inputs).last_hidden_state

    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        shrunk_dbert = shrink_mlp(model_copy, ratio=0.80)
    dt = time.perf_counter() - t0

    params_after = sum(p.numel() for p in shrunk_dbert.parameters())
    reduction_pct = (params_before - params_after) / params_before * 100.0

    print(
        f"\n[BENCHMARK] DistilBERT shrink_mlp(ratio=0.80):\n"
        f"  Surgery Duration: {dt*1000:.2f}ms\n"
        f"  Parameters Before: {params_before:,}\n"
        f"  Parameters After: {params_after:,}\n"
        f"  Actual Reduction: {reduction_pct:.2f}%"
    )
    assert params_after < params_before
    assert reduction_pct >= 2.0

    with torch.no_grad():
        new_hidden = shrunk_dbert(**distilbert_inputs).last_hidden_state

    assert new_hidden.shape == base_hidden.shape
    assert torch.isfinite(new_hidden).all()
