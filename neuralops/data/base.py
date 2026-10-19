#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/SVD_linear_Attention/graphs/contributors
"""


NeuralOps Data Base Module.

Standard data configurations, collators, and dataloader utilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader, Dataset


class TaskType(str, Enum):
    """Task type classification enumeration."""
    SINGLE_CLASSIFICATION = "single_classification"
    REGRESSION = "regression"
    PAIR_CLASSIFICATION = "pair_classification"
    LANGUAGE_MODELING = "language_modeling"


@dataclass
class DataConfig:
    """Configuration class for dataset loading and preprocessing."""
    task_type: TaskType = TaskType.SINGLE_CLASSIFICATION
    max_length: int = 128
    batch_size: int = 16
    shuffle: bool = True
    preprocessing_num_workers: int | None = None
    pad_to_multiple_of: int | None = 8


class ProductionDataCollator:
    """Production-grade data collator supporting dynamic padding to multiples."""

    def __init__(
        self,
        pad_token_id: int = 0,
        pad_to_multiple_of: int | None = 8,
        label_pad_token_id: int = -100,
    ) -> None:
        self.pad_token_id = pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of or 1
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        if not features:
            return {}

        def to_list(val: Any) -> list:
            if isinstance(val, torch.Tensor):
                return val.tolist()
            if isinstance(val, (list, tuple)):
                return list(val)
            return [val]

        max_len = max(len(to_list(f["input_ids"])) for f in features)
        if self.pad_to_multiple_of > 1:
            rem = max_len % self.pad_to_multiple_of
            if rem != 0:
                max_len += self.pad_to_multiple_of - rem

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for f in features:
            ids = to_list(f["input_ids"])
            pad_len = max_len - len(ids)
            batch_input_ids.append(ids + [self.pad_token_id] * pad_len)
            mask = to_list(f.get("attention_mask", [1] * len(ids)))
            batch_attention_mask.append(mask + [0] * pad_len)
            if "labels" in f:
                lbl = f["labels"]
                if isinstance(lbl, (list, tuple, torch.Tensor)):
                    lbl_list = to_list(lbl)
                    l_pad = max_len - len(lbl_list)
                    batch_labels.append(lbl_list + [self.label_pad_token_id] * l_pad)
                else:
                    batch_labels.append(lbl)

        batch = {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
        }
        if batch_labels:
            if isinstance(batch_labels[0], (int, float)):
                batch["labels"] = torch.tensor(batch_labels)
            else:
                batch["labels"] = torch.tensor(batch_labels, dtype=torch.long)

        return batch


def create_dataloader(
    dataset: Any,
    tokenizer: Any | None = None,
    config: DataConfig | None = None,
    collate_fn: Callable | None = None,
    **kwargs: Any,
) -> DataLoader:
    """Factory helper to create a PyTorch DataLoader from an arbitrary dataset."""
    batch_size = getattr(config, "batch_size", 16) if config is not None else kwargs.pop("batch_size", 16)
    shuffle = getattr(config, "shuffle", True) if config is not None else kwargs.pop("shuffle", False)
    if collate_fn is None:
        collate_fn = ProductionDataCollator(
            pad_to_multiple_of=getattr(config, "pad_to_multiple_of", 8) if config else 8
        )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, **kwargs)


def preprocess_function(examples: dict[str, Any], tokenizer: Any | None = None, **kwargs: Any) -> dict[str, Any]:
    """Tokenization and preprocessing helper for raw dataset batches."""
    if tokenizer is not None and "text" in examples:
        return tokenizer(examples["text"], truncation=True, **kwargs)
    return examples
