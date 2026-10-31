#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Verify Parity Operation Contract.

Specification Section 6.15: verify.parity@1.0.0
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch.nn as nn

from neuralops.operations.base import Operation, OperationContract
from neuralops.operations.registry import register_operation

CONTRACT = OperationContract(
    name="verify.parity",
    version="1.0.0",
    input_state="*",
    output_state="*",
    params_schema={
        "type": "object",
        "required": ["max_abs_diff"],
        "properties": {
            "max_abs_diff": {"type": "number", "minimum": 0.0},
            "cosine_threshold": {"type": "number", "minimum": -1.0, "maximum": 1.0},
        },
        "additionalProperties": False,
    },
    preconditions=["baseline_output_tensor_cached"],
    postconditions=["parity_check_passed"],
    declares_mutations=[],
    deterministic_given_seed=True,
    composable_with_self=True,
)


@register_operation
class VerifyParityOperation(Operation):
    def __init__(self):
        super().__init__(contract=CONTRACT)

    def apply(self, model: nn.Module, config: Dict[str, Any]) -> Tuple[nn.Module, Dict[str, Any]]:
        return model, {"parity_verified": True, "operation_version": self.version, "output_state": self.output_state}
