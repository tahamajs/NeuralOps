#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for SelfSpeculativeDecoder invariants, mathematical formulations, and model validation.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from transformers import PretrainedConfig

from neuralops.inference.speculative_decoding import (
    SelfSpeculativeDecoder,
    SpeculativeDecodeStats,
    gamma_sweep_harness,
)


class SimpleToyLM(nn.Module):
    """Deterministic toy LM for speculative decoding tests."""

    def __init__(self, vocab_size: int = 64, hidden_dim: int = 32):
        super().__init__()
        self.config = PretrainedConfig(vocab_size=vocab_size)
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = self.embed(input_ids)
        h = torch.relu(self.linear(h))
        return self.head(h)


def test_acceptance_rate_and_speedup_math():
    """Verify acceptance rate and speedup mathematical definitions."""
    # 1. Acceptance rate: accepted_drafts / total_drafts
    assert SelfSpeculativeDecoder.compute_acceptance_rate(0, 0) == 0.0
    assert SelfSpeculativeDecoder.compute_acceptance_rate(10, 20) == 0.5
    assert SelfSpeculativeDecoder.compute_acceptance_rate(15, 15) == 1.0

    with pytest.raises(ValueError, match="Draft counts must be non-negative"):
        SelfSpeculativeDecoder.compute_acceptance_rate(-1, 10)
    with pytest.raises(ValueError, match="Draft counts must be non-negative"):
        SelfSpeculativeDecoder.compute_acceptance_rate(5, -1)

    # 2. Speedup: speculative_tokens_per_sec / baseline_tokens_per_sec
    assert SelfSpeculativeDecoder.compute_speedup(50.0, 25.0) == 2.0
    assert SelfSpeculativeDecoder.compute_speedup(100.0, 50.0) == 2.0

    with pytest.raises(ValueError, match="baseline_tokens_per_sec must be positive"):
        SelfSpeculativeDecoder.compute_speedup(50.0, 0.0)
    with pytest.raises(ValueError, match="baseline_tokens_per_sec must be positive"):
        SelfSpeculativeDecoder.compute_speedup(50.0, -10.0)


def test_model_cross_check_validation():
    """Verify target_model and draft_model cross-checking."""
    target = SimpleToyLM(vocab_size=64)
    draft_valid = SimpleToyLM(vocab_size=64)
    draft_mismatch = SimpleToyLM(vocab_size=128)

    # Target model must be nn.Module
    with pytest.raises(TypeError, match="target_model must be an instance of nn.Module"):
        SelfSpeculativeDecoder(target_model="not_a_model")

    # Draft model must be nn.Module if provided
    with pytest.raises(TypeError, match="draft_model must be an instance of nn.Module"):
        SelfSpeculativeDecoder(target_model=target, draft_model="not_a_model")

    # Draft model vocab size must match target model vocab size
    with pytest.raises(ValueError, match="Draft model vocab size .* does not match"):
        SelfSpeculativeDecoder(target_model=target, draft_model=draft_mismatch)

    # Valid draft model succeeds
    decoder = SelfSpeculativeDecoder(target_model=target, draft_model=draft_valid)
    assert decoder.target_model is target
    assert decoder.draft_model is draft_valid


def test_self_speculative_decoding_exact_greedy_match():
    """Verify self-speculative generation produces identical sequence to standard autoregressive decoding."""
    torch.manual_seed(42)
    model = SimpleToyLM(vocab_size=64, hidden_dim=32).eval()
    decoder = SelfSpeculativeDecoder(target_model=model, max_draft_k=4)

    prompt = torch.randint(0, 64, (1, 6))
    max_new_tokens = 16

    # Standard greedy reference
    ref_ids = prompt.clone()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(ref_ids)[:, -1, :]
            next_t = torch.argmax(logits, dim=-1, keepdim=True)
            ref_ids = torch.cat([ref_ids, next_t], dim=-1)

    # Speculative generation
    spec_ids, stats = decoder.generate(
        input_ids=prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
    )

    assert torch.equal(ref_ids, spec_ids)
    assert stats.total_tokens_generated == max_new_tokens
    assert 0.0 <= stats.acceptance_rate <= 1.0
    assert stats.speedup_over_standard >= 1.0


def test_speculative_decoding_with_baseline_tokens_per_sec():
    """Verify speedup calculation when baseline_tokens_per_sec is provided."""
    model = SimpleToyLM(vocab_size=64, hidden_dim=32).eval()
    decoder = SelfSpeculativeDecoder(target_model=model, max_draft_k=3)
    prompt = torch.randint(0, 64, (1, 4))

    _, stats = decoder.generate(
        input_ids=prompt,
        max_new_tokens=8,
        temperature=0.0,
        baseline_tokens_per_sec=10.0,
    )

    assert isinstance(stats, SpeculativeDecodeStats)
    expected_speedup = round(stats.tokens_per_sec / 10.0, 2)
    assert stats.speedup_over_standard == expected_speedup


def test_gamma_sweep_harness():
    """Verify gamma sweep across multiple speculative horizons."""
    model = SimpleToyLM(vocab_size=64, hidden_dim=32).eval()
    prompt = torch.randint(0, 64, (1, 4))

    sweep_results = gamma_sweep_harness(
        target_model=model,
        input_ids=prompt,
        gamma_values=[1, 2, 4],
        max_new_tokens=8,
    )

    assert set(sweep_results.keys()) == {1, 2, 4}
    for gamma, stats in sweep_results.items():
        assert isinstance(stats, SpeculativeDecodeStats)
        assert stats.total_tokens_generated == 8
