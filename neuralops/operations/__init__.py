#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
NeuralOps Operations Subpackage and Contract Registry.
"""

# Eagerly import subpackages to register all 15 normative operations
import neuralops.operations.adaptation  # noqa: F401
import neuralops.operations.architecture  # noqa: F401
import neuralops.operations.attention  # noqa: F401
import neuralops.operations.export  # noqa: F401
import neuralops.operations.verify  # noqa: F401
import neuralops.operations.weights  # noqa: F401
from neuralops.operations.base import Operation, OperationContract
from neuralops.operations.errors import OperationError
from neuralops.operations.registry import get_operation, get_operation_contracts, list_operations, register_operation

__all__ = [
    "Operation",
    "OperationContract",
    "OperationError",
    "get_operation",
    "get_operation_contracts",
    "list_operations",
    "register_operation",
]
