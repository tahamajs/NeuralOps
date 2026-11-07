#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Architecture Reduce MLP Operation Contract.

Specification Section 6.7: architecture.reduce_mlp@1.0.0
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch.nn as nn

from neuralops.core.exceptions import IntegrationNotImplementedError
from neuralops.operations.base import Operation, OperationContract
from neuralops.operations.registry import register_operation

CONTRACT = OperationContract(
    name="architecture.reduce_mlp",
    version="1.0.0",
    input_state="*",
    output_state="pruned",
    params_schema={
        "type": "object",
        "required": ["ratio"],
        "properties": {
            "ratio": {"type": "number", "minimum": 0.2, "maximum": 0.95},
        },
        "additionalProperties": False,
    },
    preconditions=["has_standard_mlp_blocks"],
    postconditions=["mlp_intermediate_dim == original_dim * ratio"],
    declares_mutations=["*.mlp", "*.feed_forward"],
    deterministic_given_seed=True,
    composable_with_self=False,
)


@register_operation
class ArchitectureReduceMlpOperation(Operation):
    def __init__(self):
        super().__init__(contract=CONTRACT)

    def apply(self, model: nn.Module, config: Dict[str, Any]) -> Tuple[nn.Module, Dict[str, Any]]:
        ratio = config.get("ratio", 0.8)
        try:
            from neuralops.surgery.mlp.shrink import shrink_mlp

            shrunk = shrink_mlp(model, ratio=ratio)
            return shrunk, {"params_used": config, "operation_version": self.version, "output_state": self.output_state}
        except Exception as exc:
            raise IntegrationNotImplementedError(f"architecture.reduce_mlp failed: {exc}") from exc
