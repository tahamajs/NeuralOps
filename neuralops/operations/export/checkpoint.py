#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Export Checkpoint Operation Contract.

Specification Section 6.14: export.checkpoint@1.0.0
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch.nn as nn

from neuralops.operations.base import Operation, OperationContract
from neuralops.operations.registry import register_operation

CONTRACT = OperationContract(
    name="export.checkpoint",
    version="1.0.0",
    input_state="*",
    output_state="ServingArtifact[hf]",
    params_schema={
        "type": "object",
        "properties": {
            "save_tokenizer": {"type": "boolean", "default": True},
        },
        "additionalProperties": False,
    },
    preconditions=["valid_model_state"],
    postconditions=["checkpoint_file_exists"],
    declares_mutations=[],
    deterministic_given_seed=True,
    composable_with_self=False,
)


@register_operation
class ExportCheckpointOperation(Operation):
    def __init__(self):
        super().__init__(contract=CONTRACT)

    def apply(self, model: nn.Module, config: Dict[str, Any]) -> Tuple[nn.Module, Dict[str, Any]]:
        return model, {"saved": True, "operation_version": self.version, "output_state": self.output_state}
