"""Tests for neuralops.config.data_config."""

import pytest

from neuralops.config import DataConfig, DatasetConfig


class TestDataConfig:
    """Test DataConfig dataclass defaults and properties."""

    def test_creation(self):
        cfg = DataConfig()
        assert cfg is not None

    def test_max_seq_length(self):
        cfg = DataConfig(max_seq_length=256)
        assert cfg.max_seq_length == 256

    def test_local_glue_path(self):
        cfg = DataConfig(local_glue_path="/tmp/glue")
        assert cfg.local_glue_path == "/tmp/glue"

    def test_default_values(self):
        cfg = DataConfig()
        assert cfg.max_seq_length > 0
        assert cfg.num_proc >= 1


class TestDatasetConfig:
    """Test DatasetConfig dataclass initialization."""

    def test_creation(self):
        cfg = DatasetConfig(name="arc", path="/tmp/arc", task_type="mc", num_choices=4)
        assert cfg.name == "arc"
        assert cfg.path == "/tmp/arc"
        assert cfg.task_type == "mc"
        assert cfg.num_choices == 4
