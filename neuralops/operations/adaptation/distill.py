#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Adaptation Distill Operation Contract.

Specification Section 6.11: adaptation.distill@1.0.0
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch.nn as nn

from neuralops.core.exceptions import IntegrationNotImplementedError
from neuralops.operations.base import Operation, OperationContract
from neuralops.operations.registry import register_operation

CONTRACT = OperationContract(
    name="adaptation.distill",
    version="1.0.0",
    input_state="*",
    output_state="adapted",
    params_schema={
        "type": "object",
        "required": ["teacher_source", "steps"],
        "properties": {
            "teacher_source": {"type": "string"},
            "steps": {"type": "integer", "minimum": 10},
            "learning_rate": {"type": "number", "default": 5e-5},
            "pillars": {"type": "array", "items": {"type": "string"}},
        },
        "additionalProperties": False,
    },
    preconditions=["torch.cuda.is_available", "dataset_specified"],
    postconditions=["recovery_loss_converged"],
    declares_mutations=["*"],
    deterministic_given_seed=False,
    composable_with_self=True,
)


@register_operation
class AdaptationDistillOperation(Operation):
    def __init__(self):
        super().__init__(contract=CONTRACT)

    def apply(self, model: nn.Module, config: Dict[str, Any]) -> Tuple[nn.Module, Dict[str, Any]]:
        raise IntegrationNotImplementedError("adaptation.distill requires GPU training loop and is deferred to Phase 2.")
