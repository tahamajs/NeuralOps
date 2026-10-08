"""Tests for neuralops.data.multi_choice module."""

import json
import pandas as pd
import pytest
import torch
from unittest.mock import MagicMock

from neuralops.data.multi_choice import (
    MultiChoiceDataset,
    convert_to_multichoice_format,
    is_multichoice_dataset,
)


class TestMultiChoiceDataset:
    """Test MultiChoiceDataset parsing and tokenization."""

    def test_dataset_creation_and_tokenization(self):
        df = pd.DataFrame([
            {"question": "What is 2+2?", "choices": ["3", "4", "5"], "label": 1},
            {"question": "Color of sky?", "choices": ["Blue", "Green"], "label": 0},
        ])
        tokenizer = MagicMock()
        def mock_call(*args, **kwargs):
            return {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
            }
        tokenizer.side_effect = mock_call

        ds = MultiChoiceDataset(df, tokenizer=tokenizer, max_length=16)
        assert len(ds) == 2
        item = ds[0]
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item
        assert item["labels"].item() == 1
        # Number of choices for sample 0 is 3
        assert item["input_ids"].shape[0] == 3


class TestMultiChoiceHelpers:
    """Test multi-choice format detection and normalization."""

    def test_is_multichoice_dataset_with_choices_json(self):
        df = pd.DataFrame({
            "prompt": ["Question 1"],
            "choices_json": [json.dumps(["Option A", "Option B", "Option C"])],
        })
        is_mc, num_choices = is_multichoice_dataset(df)
        assert is_mc is True
        assert num_choices == 3

    def test_is_multichoice_dataset_false(self):
        df = pd.DataFrame({
            "text": ["Normal text sentence"],
            "label": [0],
        })
        is_mc, num_choices = is_multichoice_dataset(df)
        assert is_mc is False
        assert num_choices == 0

    def test_convert_to_multichoice_format(self):
        df = pd.DataFrame({
            "prompt": ["Sample question"],
            "choices_json": [json.dumps(["A", "B"])],
            "label": [1],
        })
        converted = convert_to_multichoice_format(df)
        assert "question" in converted.columns
        assert "choices" in converted.columns
        assert "label" in converted.columns
        assert converted.iloc[0]["label"] == 1
        assert converted.iloc[0]["choices"] == ["A", "B"]
