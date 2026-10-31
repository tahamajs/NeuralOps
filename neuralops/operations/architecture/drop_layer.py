#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Architecture Drop Layer Operation Contract.

Specification Section 6.8: architecture.drop_layer@1.0.0
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch.nn as nn

from neuralops.core.errors import IntegrationNotImplementedError
from neuralops.operations.base import Operation, OperationContract
from neuralops.operations.registry import register_operation

CONTRACT = OperationContract(
    name="architecture.drop_layer",
    version="1.0.0",
    input_state="*",
    output_state="pruned",
    params_schema={
        "type": "object",
        "required": ["drop_count"],
        "properties": {
            "drop_count": {"type": "integer", "minimum": 1},
            "strategy": {"type": "string", "enum": ["tail", "interleaved", "uniform"], "default": "tail"},
        },
        "additionalProperties": False,
    },
    preconditions=["num_layers > drop_count"],
    postconditions=["num_layers == original_num_layers - drop_count"],
    declares_mutations=["model.layers", "transformer.h", "encoder.layer"],
    deterministic_given_seed=True,
    composable_with_self=False,
)


@register_operation
class ArchitectureDropLayerOperation(Operation):
    def __init__(self):
        super().__init__(contract=CONTRACT)

    def apply(self, model: nn.Module, config: Dict[str, Any]) -> Tuple[nn.Module, Dict[str, Any]]:
        drop_count = config.get("drop_count", 1)
        try:
            from neuralops.surgery.layers.prune import prune_layers
            pruned = prune_layers(model, drop_count=drop_count)
            return pruned, {"params_used": config, "operation_version": self.version, "output_state": self.output_state}
        except Exception as exc:
            raise IntegrationNotImplementedError(f"architecture.drop_layer failed: {exc}") from exc
