#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Shared tiny, offline model factory used by the runnable examples."""

from transformers import DistilBertConfig, DistilBertForSequenceClassification


def tiny_classifier() -> DistilBertForSequenceClassification:
    """Return a small randomly initialized model; no network access is needed."""
    config = DistilBertConfig(
        vocab_size=128,
        max_position_embeddings=64,
        n_layers=1,
        n_heads=2,
        dim=32,
        hidden_dim=64,
        num_labels=2,
    )
    return DistilBertForSequenceClassification(config)


def tiny_inputs(batch_size: int = 2, sequence_length: int = 16) -> dict[str, object]:
    """Create valid token inputs for :func:`tiny_classifier`."""
    import torch

    input_ids = torch.randint(0, 128, (batch_size, sequence_length))
    return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
