"""Unified configuration module (re-export hub).

All configuration classes are now defined in dedicated modular files.
This module re-exports them for backward compatibility:

    from neuralops.config import LoRAConfig  # still works
    from neuralops.config.method_config import LoRAConfig  # canonical
"""

from dataclasses import dataclass, field

from neuralops.config.compression_config import CompressionConfig
from neuralops.config.data_config import DataConfig
from neuralops.config.experiment_config import DatasetConfig, ExperimentConfig
from neuralops.config.hardware_config import HardwareConfig
from neuralops.config.method_config import (
    DisLoRAConfig,
    FisherConfig,
    GeLoRAConfig,
    LLRDConfig,
    LoRAConfig,
    MethodConfig,
    MixoutConfig,
    R3FConfig,
    RDropConfig,
)
from neuralops.config.model_config import ModelConfig
from neuralops.config.sweep_config import SweepConfig
from neuralops.config.training_config import TrainingConfig


@dataclass
class Config:
    """Root configuration container uniting all subsystem configurations."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    method: MethodConfig = field(default_factory=MethodConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)


__all__ = [
    "LoRAConfig",
    "RDropConfig",
    "R3FConfig",
    "FisherConfig",
    "GeLoRAConfig",
    "LLRDConfig",
    "MixoutConfig",
    "DisLoRAConfig",
    "MethodConfig",
    "ModelConfig",
    "DataConfig",
    "HardwareConfig",
    "SweepConfig",
    "DatasetConfig",
    "ExperimentConfig",
    "CompressionConfig",
    "TrainingConfig",
    "Config",
]
