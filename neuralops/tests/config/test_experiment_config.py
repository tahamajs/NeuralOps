"""Tests for neuralops.config.experiment_config."""

import pytest

from neuralops.config import DatasetConfig, ExperimentConfig


class TestExperimentConfig:
    """Test ExperimentConfig dataclass fields and collections."""

    def test_creation(self):
        cfg = ExperimentConfig()
        assert cfg is not None

    def test_base_models(self):
        cfg = ExperimentConfig(base_models={"bert": "bert-base-uncased"})
        assert "bert" in cfg.base_models

    def test_svd_ranks(self):
        cfg = ExperimentConfig(svd_ranks=[16, 32, 64])
        assert cfg.svd_ranks == [16, 32, 64]

    def test_datasets(self):
        ds1 = DatasetConfig(name="sst2", path="/tmp/sst2", task_type="single")
        ds2 = DatasetConfig(name="mrpc", path="/tmp/mrpc", task_type="pair")
        cfg = ExperimentConfig(datasets=[ds1, ds2])
        assert len(cfg.datasets) == 2

    def test_output_dir(self):
        cfg = ExperimentConfig(output_dir="/tmp/results")
        assert cfg.output_dir == "/tmp/results"


class TestDatasetConfigExtended:
    """Test DatasetConfig parameters for multi-choice and classification."""

    def test_num_choices(self):
        cfg = DatasetConfig(name="arc", path="/tmp/data", task_type="mc", num_choices=4)
        assert cfg.num_choices == 4

    def test_path(self):
        cfg = DatasetConfig(name="test", path="/tmp/data", task_type="single")
        assert cfg.path == "/tmp/data"
