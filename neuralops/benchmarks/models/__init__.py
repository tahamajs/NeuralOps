#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
NeuralOps Benchmarks Models Subpackage.
"""

from neuralops.benchmarks.models.loader import (
    MODEL_ADAPTER_REGISTRY,
    BertFamilyAdapter,
    GptFamilyAdapter,
    LlamaFamilyAdapter,
    ModelAdapter,
    T5FamilyAdapter,
    cleanup_model,
    count_parameters,
    count_trainable_parameters,
    get_model_adapter,
    inspect_model_metadata,
    load_universal_model,
    register_model_adapter,
)

__all__ = [
    "ModelAdapter",
    "register_model_adapter",
    "get_model_adapter",
    "count_parameters",
    "count_trainable_parameters",
    "cleanup_model",
    "inspect_model_metadata",
    "load_universal_model",
    "BertFamilyAdapter",
    "GptFamilyAdapter",
    "LlamaFamilyAdapter",
    "T5FamilyAdapter",
    "MODEL_ADAPTER_REGISTRY",
]

from neuralops.benchmarks.models import loader
