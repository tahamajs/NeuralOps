#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Model Adaptation, Recovery, and Distillation Operations.
"""

from neuralops.operations.adaptation.distill import AdaptationDistillOperation
from neuralops.operations.adaptation.heal import AdaptationHealOperation
from neuralops.operations.adaptation.lora import AdaptationLoraOperation

__all__ = [
    "AdaptationDistillOperation",
    "AdaptationHealOperation",
    "AdaptationLoraOperation",
]
