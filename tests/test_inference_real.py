"""Tests for production inference engine features."""
import torch
import torch.nn as nn
from neuralops.inference.engine import (
    NeuralOpsInferenceEngine,
    stream_generate,
    speculative_generate,
    beam_search_generate,
    GenerationRequest,
    ContinuousBatcher,
)


class TinyLLM(nn.Module):
    def __init__(self, vocab_size: int = 64, dim: int = 32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, input_ids, **kwargs):
        h = self.embedding(input_ids)
        return self.head(h)


def test_real_inference_generate_penalties():
    model = TinyLLM()
    engine = NeuralOpsInferenceEngine(model)
    ids = torch.tensor([[1, 2, 3]])

    # Top-k & repetition penalty test
    out = engine.generate(ids, max_new_tokens=8, top_k=10, repetition_penalty=1.2)
    assert out.shape == (1, 11)


def test_real_speculative_generate():
    target = TinyLLM()
    draft = TinyLLM()
    ids = torch.tensor([[1, 2, 3]])
    out = speculative_generate(target, draft, ids, max_new_tokens=6, K=2)
    assert out.shape[1] >= 6


def test_real_beam_search_generate():
    model = TinyLLM()
    ids = torch.tensor([[1, 2, 3]])
    out = beam_search_generate(model, ids, beam_size=3, max_new_tokens=4, length_penalty=1.0)
    assert out.shape == (1, 7)


def test_real_continuous_batcher():
    model = TinyLLM()
    batcher = ContinuousBatcher(model, max_batch_size=4)
    r1 = GenerationRequest("req_1", torch.tensor([[1, 2]]), max_new_tokens=2)
    r2 = GenerationRequest("req_2", torch.tensor([[3, 4, 5]]), max_new_tokens=3)

    batcher.add_request(r1)
    batcher.add_request(r2)

    step1 = batcher.step()
    assert len(step1) == 2
    step2 = batcher.step()
    assert len(step2) == 2
    assert r1.finished  # generated 2 tokens

    step3 = batcher.step()
    assert len(step3) == 1  # only r2 remaining
    assert r2.finished
