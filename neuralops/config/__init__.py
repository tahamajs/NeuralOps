"""
Unified configuration system for the Efficient Transformer Toolkit.

Load a complete configuration from YAML + CLI:
    from neuralops.config import load_config
    cfg = load_config("config.yaml", cli_args=args)
"""

from .config import (
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
from .loader import load_config

__all__ = [
    "Config",
    "ModelConfig",
    "DataConfig",
    "TrainingConfig",
    "CompressionConfig",
    "MethodConfig",
    "HardwareConfig",
    "SweepConfig",
    "ExperimentConfig",
    "LoRAConfig",
    "RDropConfig",
    "R3FConfig",
    "FisherConfig",
    "GeLoRAConfig",
    "LLRDConfig",
    "MixoutConfig",
    "DisLoRAConfig",
    "DatasetConfig",
    "load_config",
]

# Auto-wired package members
from neuralops.config import compression_config
from neuralops.config import config
from neuralops.config import data_config
from neuralops.config import experiment_config
from neuralops.config import hardware_config
from neuralops.config import loader
from neuralops.config import method_config
from neuralops.config import model_config
from neuralops.config import sweep_config
from neuralops.config import training_config
