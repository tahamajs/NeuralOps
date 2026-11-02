#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Offline synthetic data generation utilities for NeuralOps examples.
No external network downloads or HuggingFace Hub credentials required.
"""

from __future__ import annotations

from typing import Iterator

import torch


def synthetic_tokens(
    batch_size: int = 2,
    seq_len: int = 16,
    vocab_size: int = 128,
    seed: int = 42,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """Generate a reproducible synthetic token input batch."""
    generator = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), generator=generator, device=device)
    attention_mask = torch.ones_like(input_ids, device=device)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def toy_dataloader(
    num_batches: int = 4,
    batch_size: int = 2,
    seq_len: int = 16,
    vocab_size: int = 128,
    seed: int = 42,
    device: str = "cpu",
) -> Iterator[dict[str, torch.Tensor]]:
    """Yield a sequence of offline synthetic token batches for training/evaluation loops."""
    generator = torch.Generator().manual_seed(seed)
    for _ in range(num_batches):
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), generator=generator, device=device)
        attention_mask = torch.ones_like(input_ids, device=device)
        labels = torch.randint(0, 2, (batch_size,), generator=generator, device=device)
        yield {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def toy_calibration_loader(
    num_batches: int = 2,
    batch_size: int = 2,
    seq_len: int = 16,
    vocab_size: int = 128,
    seed: int = 1337,
    device: str = "cpu",
) -> list[dict[str, torch.Tensor]]:
    """Return a list of synthetic token batches specifically for activation calibration."""
    return list(toy_dataloader(
        num_batches=num_batches,
        batch_size=batch_size,
        seq_len=seq_len,
        vocab_size=vocab_size,
        seed=seed,
        device=device,
    ))
