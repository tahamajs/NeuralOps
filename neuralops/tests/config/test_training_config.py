"""Tests for neuralops.config.training_config."""

from dataclasses import fields
import pytest

from neuralops.config import TrainingConfig


class TestTrainingConfig:
    """Test TrainingConfig dataclass hyperparameters and scheduler settings."""

    def test_creation(self):
        cfg = TrainingConfig()
        assert cfg is not None

    def test_batch_size(self):
        cfg = TrainingConfig(batch_size=16)
        assert cfg.batch_size == 16

    def test_learning_rate(self):
        cfg = TrainingConfig(learning_rate=1e-4)
        assert cfg.learning_rate == 1e-4

    def test_epochs(self):
        cfg = TrainingConfig(epochs=5)
        assert cfg.epochs == 5

    def test_optimizer_type(self):
        cfg = TrainingConfig(optimizer="adamw")
        assert cfg.optimizer == "adamw"

    def test_lr_scheduler(self):
        cfg = TrainingConfig(lr_scheduler_type="cosine")
        assert cfg.lr_scheduler_type == "cosine"

    def test_all_fields_present(self):
        cfg = TrainingConfig()
        field_names = {f.name for f in fields(cfg)}
        expected = {"batch_size", "epochs", "learning_rate"}
        assert expected.issubset(field_names)
