#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Attention Replace Operation Contract and Implementation.

Specification Section 6.1: attention.replace@1.0.0
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch.nn as nn

from neuralops.core.exceptions import IntegrationNotImplementedError
from neuralops.operations.base import Operation, OperationContract
from neuralops.operations.registry import register_operation

CONTRACT = OperationContract(
    name="attention.replace",
    version="1.0.0",
    input_state="loaded",
    output_state="attention_replaced",
    params_schema={
        "type": "object",
        "required": ["target"],
        "properties": {
            "target": {
                "type": "string",
                "enum": ["chunk_gla", "retnet", "cosformer", "decay_linear_attention", "hgrn2"],
            },
            "head_dim": {"type": "integer", "enum": [32, 64, 128]},
            "chunk_size": {"type": "integer", "default": 64},
            "gate_low_rank_dim": {"type": "integer", "default": 16},
        },
        "additionalProperties": False,
    },
    preconditions=["has_standard_attention", "model.config.hidden_size % head_dim == 0"],
    postconditions=["all_attention_modules_match"],
    declares_mutations=["*.self_attn", "*.attn"],
    deterministic_given_seed=True,
    composable_with_self=False,
    conflicts=["low_rank_projection"],
)


@register_operation
class AttentionReplaceOperation(Operation):
    def __init__(self):
        super().__init__(contract=CONTRACT)

    def apply(self, model: nn.Module, config: Dict[str, Any]) -> Tuple[nn.Module, Dict[str, Any]]:
        target = config.get("target")
        try:
            from neuralops.surgery.attention.replace import replace_attention
            transformed = replace_attention(model, attention=target)
            return transformed, {"params_used": config, "operation_version": self.version, "output_state": self.output_state}
        except Exception as exc:
            raise IntegrationNotImplementedError(
                f"attention.replace failed to execute target '{target}': {exc}"
            ) from exc
