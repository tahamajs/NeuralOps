"""Tests for neuralops.data.cached_tensors module."""

import pytest
import torch
from pathlib import Path
from torch.utils.data import TensorDataset

from neuralops.data.cached_tensors import (
    GLUE_TASK_CONFIG,
    MODEL_TOKENIZER_MAPPING,
    get_cached_tokenized_dataset,
)


class TestCachedTensorDataset:
    """Test loading and RAM caching of pre-tokenized tensor datasets."""

    def test_mappings_defined(self):
        assert "distilbert" in MODEL_TOKENIZER_MAPPING
        assert "sst2" in GLUE_TASK_CONFIG
        assert "mrpc" in GLUE_TASK_CONFIG

    def test_cache_hit_from_disk(self, tmp_path):
        cache_file = tmp_path / "sst2_distilbert_len128.pt"
        data = {
            "train_input_ids": torch.randint(0, 100, (8, 128)),
            "train_attention_mask": torch.ones(8, 128, dtype=torch.long),
            "train_labels": torch.randint(0, 2, (8,)),
            "val_input_ids": torch.randint(0, 100, (4, 128)),
            "val_attention_mask": torch.ones(4, 128, dtype=torch.long),
            "val_labels": torch.randint(0, 2, (4,)),
        }
        torch.save(data, cache_file)

        res = get_cached_tokenized_dataset(
            dataset_name="sst2",
            model_family="distilbert",
            seq_len=128,
            cache_dir=tmp_path,
        )
        assert isinstance(res, dict)
        assert "train" in res
        assert "val" in res
        assert isinstance(res["train"], TensorDataset)
        assert len(res["train"]) == 8
        assert len(res["val"]) == 4

    def test_in_memory_cache_hit(self, tmp_path):
        # Second call with same parameters should return the cached in-memory dictionary
        res1 = get_cached_tokenized_dataset(
            dataset_name="sst2",
            model_family="distilbert",
            seq_len=128,
            cache_dir=tmp_path,
        )
        res2 = get_cached_tokenized_dataset(
            dataset_name="sst2",
            model_family="distilbert",
            seq_len=128,
            cache_dir=tmp_path,
        )
        assert res1 is res2
