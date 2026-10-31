#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Architecture Structural Surgery Operations.
"""

from neuralops.operations.architecture.drop_layer import ArchitectureDropLayerOperation
from neuralops.operations.architecture.modify import ArchitectureModifyOperation
from neuralops.operations.architecture.reduce_mlp import ArchitectureReduceMlpOperation

__all__ = [
    "ArchitectureDropLayerOperation",
    "ArchitectureModifyOperation",
    "ArchitectureReduceMlpOperation",
]
