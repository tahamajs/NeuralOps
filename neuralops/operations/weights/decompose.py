#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Weights Decompose Operation Contract.

Specification Section 6.4: weights.decompose@1.0.0
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch.nn as nn

from neuralops.core.exceptions import IntegrationNotImplementedError
from neuralops.operations.base import Operation, OperationContract
from neuralops.operations.registry import register_operation

CONTRACT = OperationContract(
    name="weights.decompose",
    version="1.0.0",
    input_state="*",
    output_state="compressed",
    params_schema={
        "type": "object",
        "required": ["rank"],
        "properties": {
            "rank": {"type": "integer", "minimum": 4},
            "targets": {"type": "array", "items": {"type": "string"}},
        },
        "additionalProperties": False,
    },
    preconditions=["linear_targets_exist"],
    postconditions=["all_targets_decomposed_to_rank"],
    declares_mutations=["*"],
    deterministic_given_seed=True,
    composable_with_self=True,
)


@register_operation
class WeightsDecomposeOperation(Operation):
    def __init__(self):
        super().__init__(contract=CONTRACT)

    def apply(self, model: nn.Module, config: Dict[str, Any]) -> Tuple[nn.Module, Dict[str, Any]]:
        raise IntegrationNotImplementedError("weights.decompose is deferred to Phase 2.")
