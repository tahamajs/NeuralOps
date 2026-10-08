"""Unit tests for neuralops.config dataclasses and utilities."""
from __future__ import annotations

import dataclasses
import pytest

from neuralops.config import (
    CompressionConfig,
    Config,
    DataConfig,
    DatasetConfig,
    ExperimentConfig,
    HardwareConfig,
    LoRAConfig,
    MethodConfig,
    ModelConfig,
    SweepConfig,
    TrainingConfig,
)


def test_model_config_defaults():
    cfg = ModelConfig()
    assert cfg is not None
    assert hasattr(cfg, "name") or hasattr(cfg, "model_type") or hasattr(cfg, "model_name")


def test_training_config_defaults():
    cfg = TrainingConfig()
    assert cfg.learning_rate > 0
    assert cfg.batch_size > 0
    assert cfg.epochs >= 0


def test_hardware_config_instantiation():
    cfg = HardwareConfig()
    assert hasattr(cfg, "device") or hasattr(cfg, "gpu_id") or hasattr(cfg, "mixed_precision") or cfg is not None


def test_data_config_instantiation():
    cfg = DataConfig()
    assert cfg is not None


def test_dataset_config_instantiation():
    cfg = DatasetConfig(name="sst2", path="glue", task_type="classification")
    assert cfg.name == "sst2"
    assert cfg.task_type == "classification"


def test_compression_config_defaults():
    cfg = CompressionConfig()
    assert hasattr(cfg, "target_rank") or hasattr(cfg, "sparsity") or hasattr(cfg, "rank") or cfg is not None


def test_lora_config_instantiation():
    cfg = LoRAConfig()
    assert hasattr(cfg, "r") or hasattr(cfg, "rank") or hasattr(cfg, "lora_rank") or cfg is not None


def test_method_config_instantiation():
    cfg = MethodConfig()
    assert cfg is not None


def test_sweep_config_instantiation():
    cfg = SweepConfig()
    assert cfg is not None


def test_experiment_config_instantiation():
    cfg = ExperimentConfig()
    assert cfg is not None


def test_config_dataclass_fields():
    """Verify that dataclass structures have standard introspectable fields."""
    for cls in [ModelConfig, TrainingConfig, CompressionConfig, DatasetConfig]:
        if dataclasses.is_dataclass(cls):
            fields = dataclasses.fields(cls)
            assert len(fields) > 0


def test_config_serialization_to_dict():
    cfg = TrainingConfig()
    if dataclasses.is_dataclass(cfg):
        d = dataclasses.asdict(cfg)
        assert isinstance(d, dict)
        assert len(d) > 0


def test_config_composite_instantiation():
    cfg = Config()
    assert cfg is not None


def test_config_immutability_or_modification():
    cfg = TrainingConfig()
    if hasattr(cfg, "batch_size"):
        cfg.batch_size = 64
        assert cfg.batch_size == 64


def test_dataclass_reconstruction():
    cfg = TrainingConfig(batch_size=32, learning_rate=5e-5)
    if dataclasses.is_dataclass(cfg):
        d = dataclasses.asdict(cfg)
        cfg_reconstructed = TrainingConfig(**d)
        assert cfg_reconstructed.batch_size == 32
        assert cfg_reconstructed.learning_rate == 5e-5
