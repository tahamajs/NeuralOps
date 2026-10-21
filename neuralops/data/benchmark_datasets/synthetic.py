#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Deterministic synthetic benchmark samples."""

import random


def get_synthetic_benchmark_sample(
    dataset_name: str,
    num_samples: int = 100,
    seed: int = 0,
) -> list[str]:
    """Return reproducible plain-text samples for an offline benchmark smoke run."""
    if num_samples < 0:
        raise ValueError("num_samples must be non-negative")
    rng = random.Random(seed)
    return [
        f"Synthetic {dataset_name} sample {index}: token-{rng.randrange(1_000_000)}"
        for index in range(num_samples)
    ]


def _generate_synthetic_wikitext(count: int, seed: int = 0) -> list[dict]:
    return [
        {"text": sample}
        for sample in get_synthetic_benchmark_sample("wikitext", count, seed)
    ]
