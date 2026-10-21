#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Dataset helpers for multiple-choice language tasks."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset


def is_multichoice_dataset(data: Any) -> tuple[bool, int]:
    """Detect a table containing a choices column and report its choice count."""
    frame = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    choices_column = next((key for key in ("choices", "choices_json") if key in frame.columns), None)
    if choices_column is None or frame.empty:
        return False, 0
    try:
        choices = _parse_choices(frame.iloc[0][choices_column])
    except (TypeError, ValueError, json.JSONDecodeError):
        return False, 0
    return bool(choices), len(choices)


def convert_to_multichoice_format(data: Any) -> pd.DataFrame:
    """Normalize prompt/question and serialized choice columns into a DataFrame."""
    frame = data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    if "question" not in frame and "prompt" in frame:
        frame["question"] = frame["prompt"]
    if "choices" not in frame and "choices_json" in frame:
        frame["choices"] = frame["choices_json"].map(_parse_choices)
    if "label" not in frame and "labels" in frame:
        frame["label"] = frame["labels"]
    return frame


class MultiChoiceDataset(Dataset):
    """Tokenize every choice independently and return choice-major tensors."""

    def __init__(self, df: Any, tokenizer: Any, max_length: int = 512) -> None:
        self.frame = convert_to_multichoice_format(df).reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        if not {"question", "choices", "label"}.issubset(self.frame.columns):
            raise ValueError("data must provide question, choices, and label columns")

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.frame.iloc[index]
        question = str(row["question"])
        choices = _parse_choices(row["choices"])
        encoded = [
            self.tokenizer(
                f"{question} {choice}",
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
            for choice in choices
        ]
        ids = torch.cat([_as_tensor(item["input_ids"]) for item in encoded], dim=0)
        masks = torch.cat(
            [_as_tensor(item.get("attention_mask", torch.ones_like(item["input_ids"]))) for item in encoded],
            dim=0,
        )
        return {
            "input_ids": ids,
            "attention_mask": masks,
            "labels": torch.tensor(int(row["label"]), dtype=torch.long),
        }


def _parse_choices(value: Any) -> list[Any]:
    if isinstance(value, str):
        value = json.loads(value)
    if isinstance(value, dict):
        return list(value.get("text", value.values()))
    if isinstance(value, (tuple, list)):
        return list(value)
    raise TypeError("choices must be a list, tuple, mapping, or JSON string")


def _as_tensor(value: Any) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


__all__ = ["MultiChoiceDataset", "convert_to_multichoice_format", "is_multichoice_dataset"]
