# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unified Configuration Registry

Central configuration aggregator and re-export hub for all experiment, model, and method configurations.
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
