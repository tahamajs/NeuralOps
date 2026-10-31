#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Adaptation LoRA Operation Contract.

Specification Section 6.10: adaptation.lora@1.0.0
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch.nn as nn

from neuralops.core.errors import IntegrationNotImplementedError
from neuralops.operations.base import Operation, OperationContract
from neuralops.operations.registry import register_operation

CONTRACT = OperationContract(
    name="adaptation.lora",
    version="1.0.0",
    input_state="*",
    output_state="adapted",
    params_schema={
        "type": "object",
        "required": ["rank", "alpha"],
        "properties": {
            "rank": {"type": "integer", "minimum": 1},
            "alpha": {"type": "number", "minimum": 1.0},
            "targets": {"type": "array", "items": {"type": "string"}},
        },
        "additionalProperties": False,
    },
    preconditions=["model_weights_frozen_or_ready"],
    postconditions=["lora_adapters_attached"],
    declares_mutations=["*.lora_A", "*.lora_B"],
    deterministic_given_seed=True,
    composable_with_self=False,
)


@register_operation
class AdaptationLoraOperation(Operation):
    def __init__(self):
        super().__init__(contract=CONTRACT)

    def apply(self, model: nn.Module, config: Dict[str, Any]) -> Tuple[nn.Module, Dict[str, Any]]:
        raise IntegrationNotImplementedError("adaptation.lora is deferred to Phase 2.")
