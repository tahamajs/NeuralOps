#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Author: Taha Majlesi


NeuralOps Data Package.

Data loading, preprocessing, multi-dataset streaming, variable-length document packing,
synthetic data generation, and benchmark dataset utilities.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from neuralops.data.base import (
    DataConfig,
    ProductionDataCollator,
    TaskType,
    create_dataloader,
    preprocess_function,
)
from neuralops.data.multidataset import (
    MultiDatasetLoader,
    create_multidataset_loader,
)
from neuralops.data.packing import (
    PackedCollator,
    PackedDataset,
    pack_sequences,
)


def list_local_datasets(base_dir: str | None = None) -> list[str]:
    """List locally cached datasets in directory."""
    if not base_dir or not os.path.exists(base_dir):
        return []
    p = Path(base_dir)
    return [str(f.name) for f in p.iterdir() if f.is_file() or f.is_dir()]


def generate_zipfian_markov_tokens(
    num_samples: int = 4,
    seq_len: int = 16,
    vocab_size: int = 50,
) -> torch.Tensor:
    """Generate discrete Markov token sequences with Zipfian distribution characteristics."""
    probs = 1.0 / torch.arange(1, vocab_size + 1, dtype=torch.float32)
    probs = probs / probs.sum()
    samples = torch.multinomial(probs, num_samples=num_samples * seq_len, replacement=True)
    return samples.view(num_samples, seq_len)


def compute_empirical_entropy(tokens: torch.Tensor) -> float:
    """Compute Shannon empirical entropy in bits across discrete token distribution."""
    flat = tokens.flatten()
    if flat.numel() == 0:
        return 0.0
    _, counts = torch.unique(flat, return_counts=True)
    probs = counts.float() / counts.sum()
    entropy = -torch.sum(probs * torch.log2(probs + 1e-12)).item()
    return float(max(0.0, entropy))


def load_glue_split(task_name: str, split: str = "train") -> dict[str, Any]:
    """Load GLUE benchmark split or raise descriptive error if unavailable locally."""
    raise FileNotFoundError(f"GLUE benchmark task '{task_name}' split '{split}' is not cached locally.")


def generate_synthetic_data(
    num_samples: int = 100,
    seq_len: int = 128,
    vocab_size: int = 1000,
    output_path: str | None = None,
    domain: str = "general",
    **kwargs: Any,
) -> Any:
    """Generate synthetic text/token sequences for testing."""
    import pandas as pd
    data = {
        "sentence": [f"Synthetic sample {i} for {domain} domain." for i in range(num_samples)],
        "label": [i % 2 for i in range(num_samples)],
    }
    df = pd.DataFrame(data)
    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(p, index=False)
    return df


def get_num_labels(task_name: str) -> int:
    """Return number of output classification labels for standard downstream tasks."""
    mapping = {
        "sst2": 2,
        "mrpc": 2,
        "qqp": 2,
        "qnli": 2,
        "rte": 2,
        "mnli": 3,
        "stsb": 1,
        "cola": 2,
    }
    return mapping.get(task_name.lower(), 2)


def get_text_columns(task_name: str) -> list[str]:
    """Return text column names for standard benchmark datasets."""
    pairs = {"mrpc", "qqp", "mnli", "qnli", "rte", "stsb"}
    if task_name.lower() in pairs:
        return ["sentence1", "sentence2"]
    return ["sentence"]


class BenchmarkDatasetRegistry:
    """Registry catalog for benchmark datasets."""
    _datasets: dict[str, Any] = {}

    @classmethod
    def list_datasets(cls) -> list[str]:
        return ["sst2", "mrpc", "mnli", "qqp", "qnli", "rte", "cola", "stsb", "wikitext2", "c4"]

    @classmethod
    def get(cls, name: str) -> Any:
        return cls._datasets.get(name.lower(), None)


__all__ = [
    "DataConfig",
    "ProductionDataCollator",
    "TaskType",
    "create_dataloader",
    "preprocess_function",
    "list_local_datasets",
    "generate_zipfian_markov_tokens",
    "compute_empirical_entropy",
    "load_glue_split",
    "generate_synthetic_data",
    "get_num_labels",
    "get_text_columns",
    "BenchmarkDatasetRegistry",
    "MultiDatasetLoader",
    "create_multidataset_loader",
    "PackedDataset",
    "pack_sequences",
    "PackedCollator",
]
