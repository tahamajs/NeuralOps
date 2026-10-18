"""Tests for neuralops.data.dataset_loader module."""

import pytest
import pandas as pd
from unittest.mock import MagicMock, patch

from neuralops.data.dataset_loader import (
    ExperimentConfig,
    load_dataset,
)


class TestExperimentConfig:
    """Test ExperimentConfig fields and initialization."""

    def test_default_values(self):
        cfg = ExperimentConfig(dataset_name="sst2")
        assert cfg.dataset_name == "sst2"
        assert cfg.task_type == "single"
        assert cfg.max_seq_length == 128
        assert cfg.num_labels == 2
        assert cfg.metric == "accuracy"
        assert cfg.split_ratio == 0.1

    def test_custom_values(self):
        cfg = ExperimentConfig(
            dataset_name="custom.csv",
            dataset_path="/tmp/custom.csv",
            task_type="pair",
            max_seq_length=256,
            num_labels=3,
            metric="f1",
        )
        assert cfg.dataset_name == "custom.csv"
        assert cfg.dataset_path == "/tmp/custom.csv"
        assert cfg.task_type == "pair"
        assert cfg.max_seq_length == 256
        assert cfg.num_labels == 3
        assert cfg.metric == "f1"


class TestLoadDataset:
    """Test load_dataset routing across formats."""

    @patch("neuralops.data.dataset_loader._load_glue")
    def test_load_glue_routing(self, mock_glue):
        mock_glue.return_value = {"train": [], "validation": []}
        cfg = ExperimentConfig(dataset_name="sst2")
        result = load_dataset(cfg)
        mock_glue.assert_called_once_with(cfg)
        assert "train" in result

    @patch("neuralops.data.dataset_loader._load_synthetic")
    def test_load_synthetic_routing(self, mock_synth):
        mock_synth.return_value = {"train": [], "validation": []}
        cfg = ExperimentConfig(dataset_name="synthetic")
        result = load_dataset(cfg)
        mock_synth.assert_called_once_with(cfg)
        assert "train" in result

    @patch("neuralops.data.dataset_loader._load_custom")
    def test_load_custom_routing(self, mock_custom):
        mock_custom.return_value = {"train": []}
        cfg = ExperimentConfig(dataset_name="data.parquet")
        result = load_dataset(cfg)
        mock_custom.assert_called_once_with(cfg)
        assert "train" in result
