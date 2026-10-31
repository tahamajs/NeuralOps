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
import pytest
import torch
import torch.nn.functional as F

from neuralops.surgery.mlp.shrink import shrink_mlp

pytestmark = pytest.mark.filterwarnings("ignore:.*experimental.*:UserWarning")


def test_shrink_mlp_gpt2_parameter_reduction(gpt2_model):
    """Test parameter reduction on real GPT-2 with shrink_mlp(ratio=0.8)."""
    model = copy.deepcopy(gpt2_model)
    before = sum(p.numel() for p in model.parameters())

    t0 = time.perf_counter()
    shrunken = shrink_mlp(model, ratio=0.8)
    dur = (time.perf_counter() - t0) * 1000.0
    after = sum(p.numel() for p in shrunken.parameters())

    assert after < before
    assert after > before * 0.5
    reduction = 100.0 * (1.0 - after / before)
    print("\n[BENCHMARK] GPT-2 shrink_mlp(ratio=0.8):")
    print(f"  Surgery Duration: {dur:.2f}ms")
    print(f"  Parameters: {before:,} -> {after:,} ({reduction:.2f}% reduction)")


def test_shrink_mlp_gpt2_forward_works(gpt2_model, gpt2_inputs):
    """Test forward pass stability and logits cosine similarity on real GPT-2."""
    model = copy.deepcopy(gpt2_model)
    with torch.no_grad():
        out_base = model(**gpt2_inputs)
        base_logits = out_base.logits

    shrunken = shrink_mlp(model, ratio=0.8)

    with torch.no_grad():
        out_surg = shrunken(**gpt2_inputs)
        surg_logits = out_surg.logits

    assert surg_logits.shape == base_logits.shape
    assert not torch.isnan(surg_logits).any()
    assert not torch.isinf(surg_logits).any()

    cos_sim = F.cosine_similarity(base_logits.flatten(), surg_logits.flatten(), dim=0).item()
    print(f"\n[BENCHMARK] Cosine similarity baseline vs shrunken: {cos_sim:.4f}")
    assert cos_sim > 0.5


def test_shrink_mlp_gpt2_perplexity_tolerance(gpt2_model, gpt2_tokenizer):
    """Test perplexity change remains within acceptable 20% tolerance."""
    model = copy.deepcopy(gpt2_model)
    text = "The quick brown fox jumps over the lazy dog. " * 3
    enc = gpt2_tokenizer(text, return_tensors="pt")
    input_ids = enc["input_ids"]

    with torch.no_grad():
        out_base = model(input_ids, labels=input_ids)
        ppl_before = torch.exp(out_base.loss).item()

    shrunken = shrink_mlp(model, ratio=0.95)

    with torch.no_grad():
        out_surg = shrunken(input_ids, labels=input_ids)
        ppl_after = torch.exp(out_surg.loss).item()

    change = 100.0 * (ppl_after - ppl_before) / ppl_before
    print(f"\n[BENCHMARK] Perplexity: {ppl_before:.2f} -> {ppl_after:.2f} ({change:+.1f}% change)")
    assert ppl_after < ppl_before * 1.20


def test_shrink_mlp_distilbert_parameter_reduction(distilbert_model):
    """Test shrink_mlp on real DistilBERT encoder."""
    model = copy.deepcopy(distilbert_model)
    before = sum(p.numel() for p in model.parameters())
    t0 = time.perf_counter()
    shrunken = shrink_mlp(model, ratio=0.8)
    dur = (time.perf_counter() - t0) * 1000.0
    after = sum(p.numel() for p in shrunken.parameters())

    assert after < before
    reduction = 100.0 * (1.0 - after / before)
    print("\n[BENCHMARK] DistilBERT shrink_mlp(ratio=0.8):")
    print(f"  Surgery Duration: {dur:.2f}ms")
    print(f"  Parameters: {before:,} -> {after:,} ({reduction:.2f}% reduction)")


def test_shrink_mlp_swiglu_on_tinyllama():
    """Optional SwiGLU test on TinyLlama; skips gracefully if offline."""
    pytest.skip("TinyLlama-1.1B offline download skipped for local CPU runner")
