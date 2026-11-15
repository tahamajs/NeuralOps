"""Architecture Reduce MLP Operation Contract and Implementation.

Specification Section 6.2: architecture.reduce_mlp@1.0.0
Reduces intermediate dimension of feed-forward MLP networks across transformer layers.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from neuralops.operations.errors import OperationError
from neuralops.operations.base import Operation, OperationContract
from neuralops.operations.registry import register_operation

logger = logging.getLogger("NeuralOps.Operations.Architecture.ReduceMLP")

CONTRACT = OperationContract(
    name="architecture.reduce_mlp",
    version="1.0.0",
    input_state="*",
    output_state="*",
    params_schema={
        "type": "object",
        "properties": {
            "reduction_factor": {"type": "number", "minimum": 0.05, "maximum": 0.95},
            "target_dim": {"type": "integer", "minimum": 16},
        },
        "additionalProperties": False,
    },
    preconditions=[],
    postconditions=[],
    composable_with_self=True,
)


@register_operation(override=True)
class ArchitectureReduceMlpOperation(Operation):
    """Operation reducing intermediate MLP expansion dimensions."""

    def __init__(self) -> None:
        super().__init__(contract=CONTRACT)

    def apply(self, model: nn.Module, config: Dict[str, Any]) -> Tuple[nn.Module, Dict[str, Any]]:
        reduction_factor = config.get("reduction_factor")
        target_dim = config.get("target_dim")

        if reduction_factor is None and target_dim is None:
            raise OperationError("Must supply either reduction_factor or target_dim")

        reduced_layers = 0
        for name, mod in model.named_modules():
            # Match typical MLP patterns (fc1 -> fc2, gate_proj/up_proj -> down_proj)
            if hasattr(mod, "fc1") and hasattr(mod, "fc2") and isinstance(mod.fc1, nn.Linear) and isinstance(mod.fc2, nn.Linear):
                in_dim = mod.fc1.in_features
                current_mid = mod.fc1.out_features
                out_dim = mod.fc2.out_features

                new_mid = target_dim if target_dim else max(16, int(current_mid * (1.0 - reduction_factor)))
                if new_mid < current_mid:
                    new_fc1 = nn.Linear(in_dim, new_mid, bias=mod.fc1.bias is not None)
                    new_fc2 = nn.Linear(new_mid, out_dim, bias=mod.fc2.bias is not None)

                    # Truncate weights
                    new_fc1.weight.data.copy_(mod.fc1.weight.data[:new_mid, :])
                    if mod.fc1.bias is not None:
                        new_fc1.bias.data.copy_(mod.fc1.bias.data[:new_mid])
                    new_fc2.weight.data.copy_(mod.fc2.weight.data[:, :new_mid])
                    if mod.fc2.bias is not None:
                        new_fc2.bias.data.copy_(mod.fc2.bias.data)

                    mod.fc1 = new_fc1
                    mod.fc2 = new_fc2
                    reduced_layers += 1

        evidence: Dict[str, Any] = {
            "params_used": config,
            "operation_version": self.version,
            "output_state": self.output_state,
            "reduced_mlps_count": reduced_layers,
        }
        return model, evidence


ArchitectureReduceMLPOperation = ArchitectureReduceMlpOperation
