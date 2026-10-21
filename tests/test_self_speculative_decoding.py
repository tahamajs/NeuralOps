#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for Lossless Self-Speculative Decoding Engine.
"""

import pytest
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel

from neuralops.inference.speculative_decoding import SelfSpeculativeDecoder, _sampling_probs


def test_self_speculative_greedy_exact_distribution_parity():
    """Verify greedy speculative decoding outputs 100% bit-for-bit identical tokens to target model."""
    torch.manual_seed(42)
    config = GPT2Config(
        vocab_size=200,
        n_positions=128,
        n_embd=64,
        n_layer=2,
        n_head=2,
        pad_token_id=0,
    )
    model = GPT2LMHeadModel(config).eval()

    prompt = torch.tensor([[10, 25, 42, 99]])
    max_new = 15

    curr = prompt.clone()
    with torch.no_grad():
        for _ in range(max_new):
            logits = model(curr).logits[:, -1, :]
            next_t = torch.argmax(logits, dim=-1, keepdim=True)
            curr = torch.cat([curr, next_t], dim=-1)
    standard_tokens = curr

    draft_config = GPT2Config(
        vocab_size=200,
        n_positions=128,
        n_embd=64,
        n_layer=1,
        n_head=2,
        pad_token_id=0,
    )
    draft_model = GPT2LMHeadModel(draft_config).eval()

    decoder = SelfSpeculativeDecoder(target_model=model, draft_model=draft_model, max_draft_k=3)
    spec_tokens, stats = decoder.generate(prompt, max_new_tokens=max_new, temperature=0.0)

    assert torch.equal(spec_tokens, standard_tokens), (
        f"Speculative mismatch: {spec_tokens} vs {standard_tokens}"
    )
    assert stats.total_tokens_generated == max_new
    assert stats.total_target_forward_passes > 0
    assert stats.speedup_over_standard >= 1.0


def test_self_speculative_stochastic_sampling():
    """Verify stochastic speculative generation generates valid non-NaN token sequences."""
    torch.manual_seed(42)
    config = GPT2Config(
        vocab_size=100,
        n_positions=64,
        n_embd=32,
        n_layer=1,
        n_head=2,
        pad_token_id=0,
    )
    model = GPT2LMHeadModel(config).eval()

    prompt = torch.tensor([[5, 12, 18]])
    decoder = SelfSpeculativeDecoder(target_model=model, draft_model=model, max_draft_k=2)
    out_tokens, stats = decoder.generate(prompt, max_new_tokens=10, temperature=0.8)

    assert out_tokens.shape == (1, 13)
    assert stats.total_tokens_generated == 10
    assert not torch.isnan(out_tokens.float()).any()


def test_sampling_filters_apply_to_draft_and_target_distribution():
    probabilities = _sampling_probs(torch.tensor([[4.0, 3.0, 2.0, 1.0]]), 1.0, top_k=2, top_p=1.0)
    assert torch.count_nonzero(probabilities).item() == 2
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(1))


def test_self_speculative_decoder_rejects_multi_sequence_sampling_input():
    model = nn.Linear(4, 4)
    decoder = SelfSpeculativeDecoder(model, draft_model=model, max_draft_k=2)
    with pytest.raises(ValueError, match="requires input_ids shaped"):
        decoder.generate(torch.tensor([[1, 2], [3, 4]]), max_new_tokens=1)
