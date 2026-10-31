#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Real-model validation tests for replace_attention surgery on GPT-2 and DistilBERT.
"""

from __future__ import annotations

import copy
import time

import pytest
import torch

from neuralops.surgery.attention.replace import replace_attention


def test_replace_attention_gpt2_forward(gpt2_model, gpt2_tokenizer, short_text_inputs):
    """Test replace_attention on real GPT-2 124M with chunk_gla subquadratic attention."""
    model_copy = copy.deepcopy(gpt2_model)

    # 1. Baseline forward pass
    t0_base = time.perf_counter()
    with torch.no_grad():
        base_out = model_copy(**short_text_inputs)
        base_logits = base_out.logits
    dt_base = time.perf_counter() - t0_base

    # 2. Apply replace_attention surgery
    t0_surg = time.perf_counter()
    replaced_model = replace_attention(model_copy, attention="chunk_gla")
    dt_surg = time.perf_counter() - t0_surg

    # 3. Post-surgery forward pass
    t0_post = time.perf_counter()
    with torch.no_grad():
        new_out = replaced_model(**short_text_inputs)
        new_logits = new_out.logits
    dt_post = time.perf_counter() - t0_post

    # 4. Numerical assertions
    assert new_logits.shape == base_logits.shape
    assert not torch.isnan(new_logits).any(), "Replaced logits contain NaN"
    assert not torch.isinf(new_logits).any(), "Replaced logits contain Inf"

    mag_ratio = (new_logits.abs().mean() / base_logits.abs().mean()).item()
    print(
        f"\n[BENCHMARK] GPT-2 replace_attention('chunk_gla'):\n"
        f"  Baseline Forward: {dt_base*1000:.2f}ms\n"
        f"  Surgery Injection: {dt_surg*1000:.2f}ms\n"
        f"  Post-Surgery Forward: {dt_post*1000:.2f}ms\n"
        f"  Output Magnitude Ratio: {mag_ratio:.3f}"
    )
    assert 0.05 < mag_ratio < 20.0, f"Unexpected output scale magnitude ratio {mag_ratio}"


def test_replace_attention_gpt2_generation(gpt2_model, gpt2_tokenizer):
    """Test autoregressive token generation on real GPT-2 post-attention surgery."""
    model_copy = copy.deepcopy(gpt2_model)
    replaced_model = replace_attention(model_copy, attention="chunk_gla")

    prompt = "Hello"
    inputs = gpt2_tokenizer(prompt, return_tensors="pt")
    input_len = inputs["input_ids"].shape[1]

    t0_gen = time.perf_counter()
    with torch.no_grad():
        output_ids = replaced_model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=gpt2_tokenizer.eos_token_id,
        )
    dt_gen = time.perf_counter() - t0_gen

    generated_text = gpt2_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(
        f"\n[BENCHMARK] Autoregressive Generation (10 tokens):\n"
        f"  Time: {dt_gen*1000:.2f}ms\n"
        f"  Prompt: '{prompt}'\n"
        f"  Generated Text: '{generated_text}'"
    )
    assert len(generated_text) > len(prompt)
    assert output_ids.shape[1] == input_len + 10


def test_replace_attention_distilbert_encoder(distilbert_model, distilbert_inputs):
    """Test replace_attention on real DistilBERT 66M encoder model."""
    model_copy = copy.deepcopy(distilbert_model)

    with torch.no_grad():
        base_out = model_copy(**distilbert_inputs)
        base_hidden = base_out.last_hidden_state

    t0_surg = time.perf_counter()
    replaced_model = replace_attention(model_copy, attention="chunk_gla")
    dt_surg = time.perf_counter() - t0_surg

    with torch.no_grad():
        new_out = replaced_model(**distilbert_inputs)
        new_hidden = new_out.last_hidden_state

    assert new_hidden.shape == base_hidden.shape
    assert not torch.isnan(new_hidden).any()
    assert not torch.isinf(new_hidden).any()
    print(f"\n[BENCHMARK] DistilBERT Attention Replacement Time: {dt_surg*1000:.2f}ms")


@pytest.mark.parametrize("attention_name", ["chunk_gla", "cosformer", "decay_linear_attention"])
def test_replace_attention_multiple_mechanisms(gpt2_model, short_text_inputs, attention_name):
    """Verify that multiple attention variants inject cleanly into real GPT-2."""
    model_copy = copy.deepcopy(gpt2_model)
    t0 = time.perf_counter()
    replaced = replace_attention(model_copy, attention=attention_name)
    dt_surg = time.perf_counter() - t0

    with torch.no_grad():
        out = replaced(**short_text_inputs)
        logits = out.logits

    assert logits.shape == (1, short_text_inputs["input_ids"].shape[1], 50257)
    assert torch.isfinite(logits).all()
    print(f"\n[BENCHMARK] Mechanism '{attention_name}' injected in {dt_surg*1000:.2f}ms")
