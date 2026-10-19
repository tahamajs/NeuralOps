# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Method Configuration

Parameter specifications for low-rank SVD, LoRA, and attention compression algorithms.
"""

from dataclasses import dataclass, field


@dataclass
class LoRAConfig:
    r: int = 8
    alpha: int = 16
    dropout: float = 0.1


@dataclass
class RDropConfig:
    alpha: float = 5.0


@dataclass
class R3FConfig:
    epsilon: float = 1e-5


@dataclass
class FisherConfig:
    num_batches: int = 20
    sample_size: int = 200
    scale_max: float = 2.0


@dataclass
class GeLoRAConfig:
    margin: float = 1.0
    min_rank: int = 2
    max_rank: int = 32


@dataclass
class LLRDConfig:
    decay: float = 0.95


@dataclass
class MixoutConfig:
    p: float = 0.9


@dataclass
class DisLoRAConfig:
    beta: float = 0.5
    capacity_factor: float = 1.0


@dataclass
class MethodConfig:
    method: str = "baseline"
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    rdrop: RDropConfig = field(default_factory=RDropConfig)
    r3f: R3FConfig = field(default_factory=R3FConfig)
    fisher: FisherConfig = field(default_factory=FisherConfig)
    gelora: GeLoRAConfig = field(default_factory=GeLoRAConfig)
    llrd: LLRDConfig = field(default_factory=LLRDConfig)
    mixout: MixoutConfig = field(default_factory=MixoutConfig)
    dislora: DisLoRAConfig = field(default_factory=DisLoRAConfig)
