#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for truthful sequential generation and TGIRequestBatch dead-step removal."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from neuralops.serving.request_batcher import (
    TGIRequestBatch,
    TGIServingAdapter,
)


class DummyModel(nn.Module):
    def __init__(self, vocab_size: int = 100):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        logits = torch.zeros((batch_size, seq_len, self.vocab_size), dtype=torch.float32)
        # Predict next token as input_ids[-1] + 1
        last_tok = input_ids[:, -1]
        for b in range(batch_size):
            next_t = int((last_tok[b].item() + 1) % self.vocab_size)
            logits[b, -1, next_t] = 10.0
        return logits


def test_tgi_request_batch_step_removed():
    """Verify that dead code step() has been removed from TGIRequestBatch."""
    batch = TGIRequestBatch(max_batch_size=4, max_seq_len=64)
    assert not hasattr(batch, "step"), "TGIRequestBatch.step() should be removed"


def test_generate_sequential_execution_and_docstring():
    """Verify generate_sequential exists and has honest docstring."""
    model = DummyModel()
    adapter = TGIServingAdapter(
        model=model,
        tokenizer=None,
        max_batch_size=4,
        max_seq_len=64,
        device="cpu",
    )
    assert "Processes requests sequentially — see create_openai_serving_app for true batched serving." in (
        adapter.generate_sequential.__doc__ or ""
    )

    prompts = [
        torch.tensor([1, 2, 3], dtype=torch.long),
        torch.tensor([10, 20], dtype=torch.long),
    ]
    completions = adapter.generate_sequential(prompts, max_new_tokens=2)
    assert len(completions) == 2
    assert isinstance(completions[0], str)
    assert isinstance(completions[1], str)


def test_continuous_batching_generate_deprecation_warning():
    """Verify continuous_batching_generate emits DeprecationWarning and delegates."""
    model = DummyModel()
    adapter = TGIServingAdapter(
        model=model,
        tokenizer=None,
        max_batch_size=4,
        max_seq_len=64,
        device="cpu",
    )

    prompts = [torch.tensor([1, 2], dtype=torch.long)]
    with pytest.deprecated_call(match="continuous_batching_generate is deprecated"):
        completions = adapter.continuous_batching_generate(prompts, max_new_tokens=2)
    assert len(completions) == 1
