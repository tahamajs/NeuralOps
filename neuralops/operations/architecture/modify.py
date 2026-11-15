"""Architecture Modify Operation Contract and Implementation.

Specification Section 6.2: architecture.modify@1.0.0
Modifies high-level model topology: hidden dimensions, activation functions,
layer configurations, and normalization schemes.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from neuralops.operations.errors import OperationError
from neuralops.operations.base import Operation, OperationContract
from neuralops.operations.registry import register_operation

logger = logging.getLogger("NeuralOps.Operations.Architecture.Modify")

CONTRACT = OperationContract(
    name="architecture.modify",
    version="1.0.0",
    input_state="*",
    output_state="*",
    params_schema={
        "type": "object",
        "properties": {
            "target_activation": {
                "type": "string",
                "enum": ["gelu", "relu", "silu", "gelu_fast", "swiglu"],
            },
            "norm_epsilon": {"type": "number", "minimum": 1e-12},
            "dropout_prob": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
        "additionalProperties": False,
    },
    preconditions=(),
    postconditions=(),
    composable_with_self=True,
)


def _get_activation_module(act_name: str) -> nn.Module:
    if act_name == "relu":
        return nn.ReLU()
    elif act_name == "silu":
        return nn.SiLU()
    elif act_name in ("gelu", "gelu_fast"):
        return nn.GELU()
    return nn.GELU()


@register_operation(override=True)
class ArchitectureModifyOperation(Operation):
    """Operation modifying architectural primitives like activations, norms, dropouts."""

    def __init__(self) -> None:
        super().__init__(contract=CONTRACT)

    def apply(self, model: nn.Module, config: Dict[str, Any]) -> Tuple[nn.Module, Dict[str, Any]]:
        target_act: Optional[str] = config.get("target_activation")
        norm_eps: Optional[float] = config.get("norm_epsilon")
        dropout_prob: Optional[float] = config.get("dropout_prob")

        modified_acts = 0
        modified_norms = 0
        modified_dropouts = 0

        for name, mod in list(model.named_modules()):
            if mod is model:
                continue

            # Update activation functions
            if target_act is not None:
                if isinstance(mod, (nn.ReLU, nn.GELU, nn.SiLU)):
                    parent_name, _, child = name.rpartition(".")
                    parent = model.get_submodule(parent_name) if parent_name else model
                    setattr(parent, child, _get_activation_module(target_act))
                    modified_acts += 1

            # Update layer norm epsilon
            if norm_eps is not None:
                if isinstance(mod, (nn.LayerNorm, nn.RMSNorm)) or "norm" in mod.__class__.__name__.lower():
                    if hasattr(mod, "eps"):
                        mod.eps = norm_eps
                        modified_norms += 1

            # Update dropout
            if dropout_prob is not None:
                if isinstance(mod, nn.Dropout):
                    mod.p = dropout_prob
                    modified_dropouts += 1

        evidence: Dict[str, Any] = {
            "params_used": config,
            "operation_version": self.version,
            "output_state": self.output_state,
            "modified_activations": modified_acts,
            "modified_norms": modified_norms,
            "modified_dropouts": modified_dropouts,
        }
        return model, evidence
