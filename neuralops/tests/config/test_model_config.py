"""Tests for neuralops.config.model_config."""

import pytest

from neuralops.config import ModelConfig


class TestModelConfig:
    """Test ModelConfig dataclass fields and defaults."""

    def test_creation(self):
        cfg = ModelConfig()
        assert cfg is not None

    def test_custom_name(self):
        cfg = ModelConfig(name="bert-base-uncased")
        assert cfg.name == "bert-base-uncased"

    def test_num_labels(self):
        cfg = ModelConfig(num_labels=3)
        assert cfg.num_labels == 3

    def test_variants_field(self):
        cfg = ModelConfig()
        assert hasattr(cfg, "variants")
        assert isinstance(cfg.variants, list)
