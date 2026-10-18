"""Tests for unified config module (re-export hub)."""

import ast
from collections import defaultdict
import glob
import pytest


class TestConfigExports:
    """Test re-exports from neuralops.config root hub and submodules."""

    def test_all_configs_importable_from_root(self):
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
        )

        for cls in [
            LoRAConfig,
            HardwareConfig,
            ModelConfig,
            TrainingConfig,
            CompressionConfig,
        ]:
            assert isinstance(cls, type)

    def test_all_configs_importable_from_modular(self):
        from neuralops.config.compression_config import CompressionConfig
        from neuralops.config.data_config import DataConfig
        from neuralops.config.experiment_config import ExperimentConfig
        from neuralops.config.hardware_config import HardwareConfig
        from neuralops.config.method_config import LoRAConfig
        from neuralops.config.model_config import ModelConfig
        from neuralops.config.sweep_config import SweepConfig
        from neuralops.config.training_config import TrainingConfig

        assert LoRAConfig is not None
        assert ModelConfig is not None
        assert HardwareConfig is not None

    def test_root_config_composition(self):
        from neuralops.config import Config

        cfg = Config()
        assert hasattr(cfg, "model") and cfg.model is not None
        assert hasattr(cfg, "data") and cfg.data is not None
        assert hasattr(cfg, "training") and cfg.training is not None

    def test_no_duplicate_classes(self):
        """Verify no class is defined twice in neuralops.config module source files."""
        classes = defaultdict(list)
        for f in glob.glob("neuralops/config/*.py"):
            if ".bak" in f:
                continue
            with open(f) as fp:
                tree = ast.parse(fp.read())
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    classes[node.name].append(f)

        dupes = {k: v for k, v in classes.items() if len(v) > 1}
        assert not dupes, f"Duplicate classes: {dupes}"
