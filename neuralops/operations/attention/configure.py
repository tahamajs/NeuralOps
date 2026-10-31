#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Attention Configure Operation Contract.

Specification Section 6.2: attention.configure@1.0.0
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch.nn as nn

from neuralops.core.errors import IntegrationNotImplementedError
from neuralops.operations.base import Operation, OperationContract
from neuralops.operations.registry import register_operation

CONTRACT = OperationContract(
    name="attention.configure",
    version="1.0.0",
    input_state="*",
    output_state="*",
    params_schema={
        "type": "object",
        "properties": {
            "chunk_size": {"type": "integer", "minimum": 16},
            "clamp_min": {"type": "number"},
            "clamp_max": {"type": "number"},
        },
        "additionalProperties": False,
    },
    preconditions=["model.has_attention"],
    postconditions=["attention_parameters_configured"],
    declares_mutations=["*.attn.config"],
    deterministic_given_seed=True,
    composable_with_self=False,
)


@register_operation
class AttentionConfigureOperation(Operation):
    def __init__(self):
        super().__init__(contract=CONTRACT)

    def apply(self, model: nn.Module, config: Dict[str, Any]) -> Tuple[nn.Module, Dict[str, Any]]:
        raise IntegrationNotImplementedError("attention.configure runtime tuning is deferred to Phase 2.")
