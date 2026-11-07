#!/usr/bin/env python3
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

from typing import Any

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


class DummyConfig:
    """Canonical lightweight dummy configuration for testing and fallback initialization."""

    def __init__(
        self,
        hidden_size: int = 64,
        num_heads: int = 4,
        num_attention_heads: int | None = None,
        num_key_value_heads: int | None = None,
        intermediate_size: int | None = None,
        max_position_embeddings: int = 512,
        vocab_size: int = 1000,
        **kwargs: Any,
    ) -> None:
        self.hidden_size = int(hidden_size)
        self.d_model = int(hidden_size)
        self.dim = int(hidden_size)
        heads = int(num_attention_heads if num_attention_heads is not None else num_heads)
        self.num_attention_heads = heads
        self.num_heads = heads
        self.n_heads = heads
        self.num_key_value_heads = int(num_key_value_heads if num_key_value_heads is not None else heads)
        self.head_dim = max(1, self.hidden_size // max(1, heads))
        self.intermediate_size = int(intermediate_size if intermediate_size is not None else self.hidden_size * 4)
        self.max_position_embeddings = int(max_position_embeddings)
        self.vocab_size = int(vocab_size)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


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
    "DummyConfig",
]
