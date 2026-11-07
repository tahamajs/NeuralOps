#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Weights Prune Operation Contract.

Specification Section 6.6: weights.prune@1.0.0
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch.nn as nn

from neuralops.core.exceptions import IntegrationNotImplementedError
from neuralops.operations.base import Operation, OperationContract
from neuralops.operations.registry import register_operation

CONTRACT = OperationContract(
    name="weights.prune",
    version="1.0.0",
    input_state="*",
    output_state="pruned",
    params_schema={
        "type": "object",
        "required": ["sparsity", "pattern"],
        "properties": {
            "sparsity": {"type": "number", "minimum": 0.1, "maximum": 0.9},
            "pattern": {"type": "string", "enum": ["unstructured", "2:4", "channel"]},
        },
        "additionalProperties": False,
    },
    preconditions=["has_dense_parameters"],
    postconditions=["measured_sparsity >= sparsity"],
    declares_mutations=["*.weight"],
    deterministic_given_seed=True,
    composable_with_self=True,
)


@register_operation
class WeightsPruneOperation(Operation):
    def __init__(self):
        super().__init__(contract=CONTRACT)

    def apply(self, model: nn.Module, config: Dict[str, Any]) -> Tuple[nn.Module, Dict[str, Any]]:
        raise IntegrationNotImplementedError("weights.prune is deferred to Phase 2.")
