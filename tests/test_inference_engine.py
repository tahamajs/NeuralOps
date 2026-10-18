"""Tests for Phase 6 Inference engine features."""
import torch
import torch.nn as nn
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
