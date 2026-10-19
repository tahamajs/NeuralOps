#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Trainer Factory

Factory interface for instantiating specialized trainers based on model and training configurations.
"""

from typing import Any

from neuralops.core.registry import Registry

from .hf_trainer import HuggingFaceTrainer

TRAINER_FACTORY_REGISTRY = Registry("trainer_factories")


def get_trainer(key: str = "hf") -> Any:
    """Return a trainer class (not instance)."""
    trainer_cls = TRAINER_FACTORY_REGISTRY.get(key)
    if trainer_cls is None:
        return HuggingFaceTrainer
    return trainer_cls
