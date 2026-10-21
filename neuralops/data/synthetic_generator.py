#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Deterministic synthetic token and task datasets for offline examples."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


def generate_zipfian_tokens(
    num_samples: int,
    seq_len: int,
    vocab_size: int,
    seed: int = 0,
) -> torch.Tensor:
    """Generate token IDs sampled from a Zipf-like rank-frequency distribution."""
    if num_samples < 0 or seq_len <= 0 or vocab_size <= 0:
        raise ValueError("num_samples must be non-negative; seq_len and vocab_size must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    ranks = torch.arange(1, vocab_size + 1, dtype=torch.float64)
    probabilities = ranks.reciprocal()
    probabilities /= probabilities.sum()
    return torch.multinomial(
        probabilities,
        num_samples * seq_len,
        replacement=True,
        generator=generator,
    ).reshape(num_samples, seq_len).long()


class SyntheticClassificationDataset(Dataset):
    """Return deterministic synthetic token sequences with class labels."""

    def __init__(self, num_samples: int, seq_len: int, vocab_size: int, num_classes: int = 2, seed: int = 0) -> None:
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        self.input_ids = generate_zipfian_tokens(num_samples, seq_len, vocab_size, seed)
        self.labels = torch.arange(num_samples, dtype=torch.long) % num_classes

    def __len__(self) -> int:
        return self.input_ids.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        input_ids = self.input_ids[index]
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "labels": self.labels[index],
        }


class SyntheticPerplexityDataset(Dataset):
    """Return deterministic causal-LM token and next-token label sequences."""

    def __init__(self, num_samples: int, seq_len: int, vocab_size: int, seed: int = 0) -> None:
        self.input_ids = generate_zipfian_tokens(num_samples, seq_len, vocab_size, seed)

    def __len__(self) -> int:
        return self.input_ids.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        tokens = self.input_ids[index]
        return {"input_ids": tokens, "attention_mask": torch.ones_like(tokens), "labels": tokens.clone()}


__all__ = ["SyntheticClassificationDataset", "SyntheticPerplexityDataset", "generate_zipfian_tokens"]
