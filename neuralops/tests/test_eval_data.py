"""Tests for eval_data module."""

import torch
from neuralops.evaluation.eval_data import (
    _generate_synthetic_batch,
    get_dataset_batch,
)


def test_generate_synthetic_batch():
    batch = _generate_synthetic_batch("sst2", batch_size=4, seq_len=16, vocab_size=1000)
    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "labels" in batch
    assert batch["input_ids"].shape == (4, 16)
    assert batch["attention_mask"].shape == (4, 16)
    assert batch["labels"].shape == (4,)


def test_generate_synthetic_batch_lm():
    batch = _generate_synthetic_batch("wikitext2", batch_size=4, seq_len=16, vocab_size=1000)
    assert batch["labels"].shape == (4, 16)


def test_get_dataset_batch_synthetic():
    batch = get_dataset_batch("synthetic", "bert", batch_size=2, seq_len=8, allow_synthetic=True)
    assert batch["input_ids"].shape == (2, 8)
