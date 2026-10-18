"""Tests for eval_utils module."""

import torch
import torch.nn as nn
from neuralops.evaluation.eval_utils import (
    _filter_batch,
    _extract_logits_and_loss,
    _evict_if_needed,
)


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, input_ids, labels=None):
        out = self.fc(input_ids.float())
        loss = None
        if labels is not None:
            loss = out.sum()
        return type("Output", (), {"logits": out, "loss": loss})()


def test_filter_batch():
    model = DummyModel()
    batch = {
        "input_ids": torch.zeros((2, 10)),
        "invalid_key": torch.zeros((2, 5)),
        "labels": torch.ones(2),
    }
    filtered = _filter_batch(batch, "bert", model)
    assert "input_ids" in filtered
    assert "labels" in filtered
    assert "invalid_key" not in filtered


def test_extract_logits_and_loss():
    model = DummyModel()
    input_ids = torch.zeros((2, 10))
    labels = torch.ones(2, dtype=torch.long)
    out = model(input_ids, labels=labels)

    logits, loss = _extract_logits_and_loss(out, num_labels=2, labels=labels)
    assert logits is not None
    assert loss is not None
    assert logits.shape == (2, 2)


def test_evict_if_needed():
    cache = {i: i for i in range(20)}
    _evict_if_needed(cache, max_size=10)
    assert len(cache) == 10
    assert 0 not in cache
    assert 19 in cache
