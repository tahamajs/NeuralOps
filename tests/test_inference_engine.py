#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for Phase 6 Inference engine features."""
import torch
import torch.nn as nn
import pytest
from neuralops.inference.engine import (
    NeuralOpsInferenceEngine,
    stream_generate,
    speculative_generate,
    beam_search_generate,
    ContinuousBatcher,
)


class DummyLLM(nn.Module):
    def __init__(self, vocab_size: int = 32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 16)
        self.head = nn.Linear(16, vocab_size)

    def forward(self, input_ids):
        h = self.embedding(input_ids)
        return self.head(h)


def test_inference_engine_generate():
    model = DummyLLM()
    engine = NeuralOpsInferenceEngine(model)
    ids = torch.tensor([[1, 2, 3]])
    out = engine.generate(ids, max_new_tokens=4)
    assert out.shape == (1, 7)


def test_stream_generate():
    model = DummyLLM()
    ids = torch.tensor([[1, 2, 3]])
    tokens = list(stream_generate(model, ids, max_new_tokens=4))
    assert len(tokens) == 4
    assert all(isinstance(t, int) for t in tokens)


def test_speculative_generate():
    target = DummyLLM()
    draft = DummyLLM()
    ids = torch.tensor([[1, 2, 3]])
    out = speculative_generate(target, draft, ids, gamma=2, max_new_tokens=4)
    assert out.shape[1] >= 4


def test_beam_search_generate():
    model = DummyLLM()
    ids = torch.tensor([[1, 2, 3]])
    out = beam_search_generate(model, ids, beam_width=2, max_new_tokens=3)
    assert out.shape == (1, 6)


def test_continuous_batcher():
    model = DummyLLM()
    batcher = ContinuousBatcher(max_batch_size=4)
    batcher.add_request("req1", torch.tensor([[1, 2]]))
    batcher.add_request("req2", torch.tensor([[3, 4]]))

    res = batcher.step(model)
    assert len(res) == 2
    assert res[0]["id"] == "req1"
    assert res[1]["id"] == "req2"


def test_generation_does_not_retry_after_model_internal_type_error():
    class BrokenModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, input_ids):
            self.calls += 1
            raise TypeError("internal model error")

    model = BrokenModel()
    with torch.no_grad(), pytest.raises(TypeError, match="internal model error"):
        NeuralOpsInferenceEngine(model).generate(torch.tensor([[1]]), max_new_tokens=1, use_cache=True)
    assert model.calls == 1


def test_generation_samples_when_temperature_is_positive_without_filters(monkeypatch):
    calls = []
    original_multinomial = torch.multinomial

    def record_multinomial(*args, **kwargs):
        calls.append(True)
        return original_multinomial(*args, **kwargs)

    monkeypatch.setattr(torch, "multinomial", record_multinomial)
    model = DummyLLM()
    NeuralOpsInferenceEngine(model).generate(torch.tensor([[1]]), max_new_tokens=1, temperature=0.7)
    assert calls


def test_continuous_batcher_selects_last_unpadded_position():
    class PositionModel(nn.Module):
        def forward(self, input_ids, attention_mask=None):
            batch_size, seq_len = input_ids.shape
            logits = torch.zeros(batch_size, seq_len, 16, device=input_ids.device)
            logits.scatter_(2, ((input_ids + 1) % 16).unsqueeze(-1), 1.0)
            return logits

    batcher = ContinuousBatcher(max_batch_size=2, max_seq_len=8)
    batcher.add_request("short", torch.tensor([[1]]), max_new_tokens=1)
    batcher.add_request("long", torch.tensor([[3, 4]]), max_new_tokens=1)

    results = batcher.step(PositionModel())

    assert [result["token"] for result in results] == [2, 5]


def test_continuous_batcher_rejects_duplicate_request_ids():
    batcher = ContinuousBatcher(max_seq_len=8)
    batcher.add_request("same", torch.tensor([[1]]), max_new_tokens=1)
    with pytest.raises(ValueError, match="already queued"):
        batcher.add_request("same", torch.tensor([[2]]), max_new_tokens=1)
