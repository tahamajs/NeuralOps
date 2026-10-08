"""Unit tests for dataset utilities, discovery, and synthetic data generation."""

import numpy as np
import pytest
import torch

from neuralops.data import (
    GLUE_TASK_KEYS,
    compute_empirical_entropy,
    generate_zipfian_markov_tokens,
    get_metric_name,
    get_num_labels,
    get_num_labels_and_is_regression,
    get_text_columns,
    list_local_datasets,
)


class TestDataDiscovery:
    """Test local dataset directory scanning and task mapping."""

    def test_list_local_datasets(self, tmp_path):
        # Create dummy cache directory structure
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        (cache_dir / "wikitext2").mkdir()
        (cache_dir / "glue_sst2").mkdir()

        found = list_local_datasets(base_dir=str(tmp_path))
        assert isinstance(found, list)

    def test_glue_task_keys_mapping(self):
        assert "sst2" in GLUE_TASK_KEYS
        assert "mrpc" in GLUE_TASK_KEYS
        assert "qqp" in GLUE_TASK_KEYS
        assert GLUE_TASK_KEYS["sst2"] == ("sentence", None)

    def test_task_helpers(self):
        assert get_metric_name("sst2") in ("accuracy", "acc")
        num_labels, is_regression = get_num_labels_and_is_regression("stsb")
        assert num_labels == 1
        assert is_regression is True

        labels = get_num_labels("mnli")
        assert labels == 3

        cols = get_text_columns("mrpc")
        assert len(cols) == 2


class TestSyntheticDataGeneration:
    """Test synthetic token distribution generators and entropy calculations."""

    def test_generate_zipfian_markov_tokens(self):
        tokens = generate_zipfian_markov_tokens(
            num_samples=10,
            seq_len=64,
            vocab_size=100,
            zipf_param=1.5,
        )
        assert isinstance(tokens, (torch.Tensor, np.ndarray))
        assert tokens.shape == (10, 64)

    def test_compute_empirical_entropy(self):
        # Uniform or distributed sequence
        tokens = torch.randint(0, 50, (5, 32))
        entropy = compute_empirical_entropy(tokens, vocab_size=50)
        assert isinstance(entropy, float)
        assert entropy >= 0.0
