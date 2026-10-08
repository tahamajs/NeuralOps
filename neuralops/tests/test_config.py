"""Tests for NeuralOps configuration system and loaders."""

import argparse
from pathlib import Path
import pytest
import yaml

from neuralops.config import (
    CompressionConfig,
    Config,
    DataConfig,
    DatasetConfig,
    DisLoRAConfig,
    ExperimentConfig,
    FisherConfig,
    GeLoRAConfig,
    HardwareConfig,
    LLRDConfig,
    LoRAConfig,
    MethodConfig,
    MixoutConfig,
    ModelConfig,
    R3FConfig,
    RDropConfig,
    SweepConfig,
    TrainingConfig,
    load_config,
)


class TestConfigInstantiation:
    """Test default values and nested composition of Config dataclasses."""

    def test_default_root_config(self):
        cfg = Config()
        assert isinstance(cfg.model, ModelConfig)
        assert isinstance(cfg.data, DataConfig)
        assert isinstance(cfg.training, TrainingConfig)
        assert isinstance(cfg.compression, CompressionConfig)
        assert isinstance(cfg.method, MethodConfig)
        assert isinstance(cfg.hardware, HardwareConfig)
        assert isinstance(cfg.sweep, SweepConfig)
        assert isinstance(cfg.experiment, ExperimentConfig)

    def test_method_config_subconfigs(self):
        method_cfg = MethodConfig()
        assert isinstance(method_cfg.lora, LoRAConfig)
        assert isinstance(method_cfg.rdrop, RDropConfig)
        assert isinstance(method_cfg.r3f, R3FConfig)
        assert isinstance(method_cfg.fisher, FisherConfig)
        assert isinstance(method_cfg.gelora, GeLoRAConfig)
        assert isinstance(method_cfg.llrd, LLRDConfig)
        assert isinstance(method_cfg.mixout, MixoutConfig)
        assert isinstance(method_cfg.dislora, DisLoRAConfig)

    def test_lora_config_defaults(self):
        lora = LoRAConfig()
        assert lora.r == 8
        assert lora.alpha == 16
        assert lora.dropout == 0.1

    def test_hardware_config_defaults(self):
        hw = HardwareConfig()
        assert hw.device in ("auto", "cuda", "cpu", "mps")
        assert hw.force_cpu_for_svd is True


class TestConfigLoader:
    """Test loading and overriding configuration from YAML and CLI args."""

    def test_load_config_defaults_without_file(self):
        cfg = load_config()
        assert isinstance(cfg, Config)
        assert cfg.model.name == "distilbert-base-uncased"

    def test_load_config_from_yaml(self, tmp_path):
        yaml_content = {
            "model": {"name": "roberta-base"},
            "training": {"learning_rate": 1e-4, "epochs": 5},
            "hardware": {"device": "cpu"},
        }
        yaml_path = tmp_path / "test_config.yaml"
        yaml_path.write_text(yaml.dump(yaml_content))

        cfg = load_config(yaml_path=str(yaml_path))
        assert cfg.model.name == "roberta-base"
        assert cfg.training.learning_rate == 1e-4
        assert cfg.training.epochs == 5
        assert cfg.hardware.device == "cpu"

    def test_load_config_cli_overrides(self, tmp_path):
        yaml_content = {
            "model": {"name": "gpt2"},
            "training": {"epochs": 3},
        }
        yaml_path = tmp_path / "test_config.yaml"
        yaml_path.write_text(yaml.dump(yaml_content))

        # CLI overrides
        parser = argparse.ArgumentParser()
        parser.add_argument("--epochs", type=int, default=10)
        cli_args = parser.parse_args(["--epochs", "10"])

        cfg = load_config(yaml_path=str(yaml_path), cli_args=cli_args)
        assert isinstance(cfg, Config)
