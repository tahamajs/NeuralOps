#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Real-model validation tests for prune_layers surgery on GPT-2.
"""

from __future__ import annotations

import copy
import pytest
import torch
import torch.nn.functional as F

from neuralops.surgery.layers.prune import prune_layers
import neuralops.surgery.layers.prune as prune_module


def test_prune_layers_gpt2_layer_count(gpt2_model):
    """Verify pruning 12 -> 10 layers on real GPT-2."""
    model = copy.deepcopy(gpt2_model)
    assert len(model.transformer.h) == 12
    assert model.config.num_hidden_layers == 12

    with pytest.warns(UserWarning, match="experimental"):
        pruned = prune_layers(model, drop_count=2)

    assert len(pruned.transformer.h) == 10
    assert pruned.config.num_hidden_layers == 10


def test_prune_layers_gpt2_forward_works(gpt2_model, gpt2_inputs):
    """Verify forward pass on 10 layers and compute cosine similarity."""
    model = copy.deepcopy(gpt2_model)
    model.eval()

    with torch.no_grad():
        orig_logits = model(**gpt2_inputs).logits

    with pytest.warns(UserWarning, match="experimental"):
        pruned = prune_layers(model, drop_count=2)
    pruned.eval()

    with torch.no_grad():
        pruned_logits = pruned(**gpt2_inputs).logits

    assert pruned_logits.shape == orig_logits.shape
    assert not torch.isnan(pruned_logits).any()

    # Compute cosine similarity across flattened logits
    cos_sim = F.cosine_similarity(
        orig_logits.flatten().unsqueeze(0),
        pruned_logits.flatten().unsqueeze(0),
    ).item()

    assert not torch.isnan(torch.tensor(cos_sim))
    assert -1.0 <= cos_sim <= 1.0
    assert cos_sim > 0.3


def test_prune_layers_gpt2_generation_works(gpt2_model, gpt2_tokenizer, gpt2_inputs):
    """Verify text generation produces valid tokens on pruned model."""
    model = copy.deepcopy(gpt2_model)
    with pytest.warns(UserWarning, match="experimental"):
        pruned = prune_layers(model, drop_count=2)
    pruned.eval()

    input_ids = gpt2_inputs["input_ids"]
    prompt_len = input_ids.shape[1]
    new_tokens = 10

    with torch.no_grad():
        output_ids = pruned.generate(
            input_ids,
            attention_mask=gpt2_inputs.get("attention_mask"),
            max_new_tokens=new_tokens,
            pad_token_id=gpt2_tokenizer.eos_token_id,
            do_sample=False,
        )

    assert output_ids.shape[1] == prompt_len + new_tokens
    generated_text = gpt2_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    assert isinstance(generated_text, str)
    assert len(generated_text) > 0


def test_prune_layers_docstring_documents_limitation():
    """Assert docstring explicitly documents distillation/healing and GPU requirements."""
    func_doc = prune_layers.__doc__
    mod_doc = prune_module.__doc__

    assert func_doc is not None
    assert "healing" in func_doc or "GPU" in func_doc
    assert mod_doc is not None
    assert "healing" in mod_doc or "GPU" in mod_doc
