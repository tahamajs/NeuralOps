#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Adaptation Heal Operation Contract.

Specification Section 6.12: adaptation.heal@1.0.0
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch.nn as nn

from neuralops.core.errors import IntegrationNotImplementedError
from neuralops.operations.base import Operation, OperationContract
from neuralops.operations.registry import register_operation

CONTRACT = OperationContract(
    name="adaptation.heal",
    version="1.0.0",
    input_state="*",
    output_state="adapted",
    params_schema={
        "type": "object",
        "required": ["calibration_tokens"],
        "properties": {
            "calibration_tokens": {"type": "integer", "minimum": 1024},
            "mode": {"type": "string", "enum": ["norm_only", "bias_only", "full_quick"]},
        },
        "additionalProperties": False,
    },
    preconditions=["model.is_mutated"],
    postconditions=["model.is_calibrated"],
    declares_mutations=["*.ln_*", "*.norm*"],
    deterministic_given_seed=True,
    composable_with_self=False,
)


@register_operation
class AdaptationHealOperation(Operation):
    def __init__(self):
        super().__init__(contract=CONTRACT)

    def apply(self, model: nn.Module, config: Dict[str, Any]) -> Tuple[nn.Module, Dict[str, Any]]:
        raise IntegrationNotImplementedError("adaptation.heal is deferred to Phase 2.")
