"""Tests for neuralops.config.sweep_config."""

import pytest

from neuralops.config import SweepConfig


class TestSweepConfig:
    """Test SweepConfig dataclass defaults and search space settings."""

    def test_creation(self):
        cfg = SweepConfig()
        assert cfg is not None

    def test_name(self):
        cfg = SweepConfig(name="rank_sweep")
        assert cfg.name == "rank_sweep"

    def test_algorithm(self):
        cfg = SweepConfig(algorithm="bayes")
        assert cfg.algorithm == "bayes"

    def test_direction(self):
        cfg = SweepConfig(direction="maximize")
        assert cfg.direction == "maximize"

    def test_search_space(self):
        cfg = SweepConfig(search_space={"rank": [16, 32, 64]})
        assert "rank" in cfg.search_space
        assert cfg.search_space["rank"] == [16, 32, 64]
