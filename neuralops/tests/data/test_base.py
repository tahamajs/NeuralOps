"""Tests for neuralops.data.base module."""

import pytest
import torch
from unittest.mock import MagicMock
from torch.utils.data import DataLoader, Dataset

from neuralops.data.base import (
    DataConfig,
    ProductionDataCollator,
    TaskType,
    create_dataloader,
    preprocess_function,
)


class DummyDataset:
    """Mock huggingface-style dataset."""

    def __init__(self, items):
        self.items = items
        self.column_names = list(items[0].keys()) if items else []

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

    def set_format(self, type, columns):
        pass


class TestDataConfig:
    """Test DataConfig configuration initialization and validation."""

    def test_default_config(self):
        cfg = DataConfig()
        assert cfg.task_type == TaskType.SINGLE_CLASSIFICATION
        assert cfg.max_length == 128
        assert cfg.batch_size == 16
        assert cfg.shuffle is True

    def test_config_with_values(self):
        cfg = DataConfig(
            task_type=TaskType.REGRESSION,
            max_length=256,
            batch_size=32,
            shuffle=False,
        )
        assert cfg.task_type == TaskType.REGRESSION
        assert cfg.max_length == 256
        assert cfg.batch_size == 32
        assert cfg.shuffle is False


class TestTaskType:
    """Test TaskType enum values."""

    def test_enum_members(self):
        assert TaskType.SINGLE_CLASSIFICATION.value == "single_classification"
        assert TaskType.REGRESSION.value == "regression"
        assert TaskType.PAIR_CLASSIFICATION.value == "pair_classification"
        assert TaskType.LANGUAGE_MODELING.value == "language_modeling"


class TestProductionDataCollator:
    """Test production data batch collator with padding."""

    def test_collate_features(self):
        collator = ProductionDataCollator(pad_token_id=0, pad_to_multiple_of=8)
        features = [
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": 1},
            {"input_ids": [1, 2, 3, 4, 5], "attention_mask": [1, 1, 1, 1, 1], "labels": 0},
        ]
        batch = collator(features)
        assert "input_ids" in batch
        assert batch["input_ids"].shape[0] == 2
        # padded to multiple of 8: length should be 8
        assert batch["input_ids"].shape[1] == 8
        assert batch["labels"].shape[0] == 2

    def test_empty_features(self):
        collator = ProductionDataCollator()
        assert collator([]) == {}


class TestCreateDataloader:
    """Test DataLoader creation from tokenized datasets."""

    def test_create_dataloader_tokenized(self):
        items = [
            {"input_ids": torch.tensor([1, 2, 3, 4]), "attention_mask": torch.tensor([1, 1, 1, 1]), "labels": 0},
            {"input_ids": torch.tensor([5, 6, 7, 8]), "attention_mask": torch.tensor([1, 1, 1, 1]), "labels": 1},
        ]
        ds = DummyDataset(items)
        cfg = DataConfig(batch_size=2, shuffle=False)
        loader = create_dataloader(ds, tokenizer=None, config=cfg)
        assert isinstance(loader, DataLoader)
        batch = next(iter(loader))
        assert batch["input_ids"].shape[0] == 2
