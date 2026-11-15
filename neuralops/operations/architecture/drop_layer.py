"""Architecture Drop Layer Operation Contract and Implementation.

Specification Section 6.2: architecture.drop_layer@1.0.0
Prunes / drops redundant transformer layers or residual blocks cleanly.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch.nn as nn

from neuralops.operations.errors import OperationError
from neuralops.operations.base import Operation, OperationContract
from neuralops.operations.registry import register_operation

logger = logging.getLogger("NeuralOps.Operations.Architecture.DropLayer")

CONTRACT = OperationContract(
    name="architecture.drop_layer",
    version="1.0.0",
    input_state="*",
    output_state="*",
    params_schema={
        "type": "object",
        "properties": {
            "layer_indices": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 1,
            },
            "layer_stack_attr": {"type": "string"},
        },
        "required": ["layer_indices"],
        "additionalProperties": False,
    },
    preconditions=(),
    postconditions=(),
    composable_with_self=True,
)


@register_operation(override=True)
class ArchitectureDropLayerOperation(Operation):
    """Operation dropping layers from a transformer layer stack (ModuleList)."""

    def __init__(self) -> None:
        super().__init__(contract=CONTRACT)

    def apply(self, model: nn.Module, config: Dict[str, Any]) -> Tuple[nn.Module, Dict[str, Any]]:
        indices_to_drop = set(config.get("layer_indices", []))
        stack_attr = config.get("layer_stack_attr")

        # Find ModuleList that contains transformer layers
        target_stack: Optional[Tuple[str, nn.ModuleList]] = None
        if stack_attr:
            mod = model.get_submodule(stack_attr) if hasattr(model, stack_attr) else None
            if isinstance(mod, nn.ModuleList):
                target_stack = (stack_attr, mod)

        if target_stack is None:
            for name, mod in model.named_modules():
                if isinstance(mod, nn.ModuleList) and len(mod) > 1:
                    target_stack = (name, mod)
                    break

        if target_stack is None:
            raise OperationError("No candidate nn.ModuleList found for dropping layers.")

        stack_name, module_list = target_stack
        orig_count = len(module_list)

        kept_modules = [m for idx, m in enumerate(module_list) if idx not in indices_to_drop]
        if not kept_modules:
            raise OperationError("Cannot drop all layers in ModuleList; at least 1 must remain.")

        new_module_list = nn.ModuleList(kept_modules)
        parent_name, _, child = stack_name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child, new_module_list)

        evidence: Dict[str, Any] = {
            "params_used": config,
            "operation_version": self.version,
            "output_state": self.output_state,
            "original_layer_count": orig_count,
            "remaining_layer_count": len(kept_modules),
            "dropped_indices": sorted(list(indices_to_drop)),
            "stack_path": stack_name,
        }
        return model, evidence
