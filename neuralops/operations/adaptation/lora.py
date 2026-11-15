#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Adaptation LoRA Operation Contract.

Specification Section 6.10: adaptation.lora@1.0.0
Connects to neuralops.core.peft.lora adapter injection.
"""

from __future__ import annotations

import fnmatch
import logging
from typing import Any, Dict, List, Optional, Tuple

import torch.nn as nn

from neuralops.core.model_utils import replace_module
from neuralops.core.peft.lora import LoRALinear
from neuralops.operations.base import Operation, OperationContract
from neuralops.operations.errors import OperationExecutionError
from neuralops.operations.registry import register_operation

logger = logging.getLogger("NeuralOps.Operations.Adaptation.LoRA")

CONTRACT = OperationContract(
    name="adaptation.lora",
    version="1.0.0",
    input_state="*",
    output_state="adapted",
    params_schema={
        "type": "object",
        "required": ["rank", "alpha"],
        "properties": {
            "rank": {"type": "integer", "minimum": 1},
            "alpha": {"type": "number", "minimum": 1.0},
            "targets": {"type": "array", "items": {"type": "string"}},
        },
        "additionalProperties": False,
    },
    preconditions=["model_weights_frozen_or_ready"],
    postconditions=["lora_adapters_attached"],
    declares_mutations=["*.lora_A", "*.lora_B"],
    deterministic_given_seed=True,
    composable_with_self=False,
)


@register_operation(override=True)
class AdaptationLoraOperation(Operation):
    """Operation for injecting LoRA adapters into model linear layers."""

    def __init__(self) -> None:
        """Initialize AdaptationLoraOperation with canonical contract."""
        super().__init__(contract=CONTRACT)

    def apply(
        self,
        model: nn.Module,
        config: Dict[str, Any],
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """Inject LoRA adapters into target linear layers.

        Args:
            model: The neural network module to adapt.
            config: Configuration dictionary specifying rank, alpha, and targets.

        Returns:
            Tuple of (adapted_model, evidence_dict).

        Raises:
            OperationExecutionError: If validation or injection fails.
        """
        try:
            rank = int(config.get("rank", 16))
            alpha = float(config.get("alpha", 32.0))
            dropout = float(config.get("dropout", 0.0))
            targets: Optional[List[str]] = config.get("targets", None)
            freeze_base = bool(config.get("freeze_base", True))

            if rank < 1:
                raise ValueError(f"rank must be >= 1, got {rank}")
            if alpha < 1.0:
                raise ValueError(f"alpha must be >= 1.0, got {alpha}")

            adapted_layers: List[str] = []
            named_modules = list(model.named_modules())

            for name, module in named_modules:
                if not isinstance(module, nn.Linear) or isinstance(module, LoRALinear):
                    continue

                # Filter by targets if provided
                if targets is not None:
                    matched = any(fnmatch.fnmatch(name, pat) or pat in name for pat in targets)
                    if not matched:
                        continue

                lora_layer = LoRALinear(
                    module,
                    r=rank,
                    lora_alpha=alpha,
                    lora_dropout=dropout,
                    freeze_base=freeze_base,
                )
                replace_module(model, name, lora_layer)
                adapted_layers.append(name)

            evidence: Dict[str, Any] = {
                "operation": "adaptation.lora",
                "rank": rank,
                "alpha": alpha,
                "targets": targets,
                "adapted_count": len(adapted_layers),
                "adapted_layers": adapted_layers,
            }
            logger.info(
                "Successfully injected %d LoRA adapters (rank=%d, alpha=%.1f)",
                len(adapted_layers),
                rank,
                alpha,
            )
            return model, evidence

        except Exception as exc:
            if isinstance(exc, OperationExecutionError):
                raise
            raise OperationExecutionError(f"Failed to execute adaptation.lora: {exc}") from exc
