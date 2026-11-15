#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Weights Decompose Operation Contract.

Specification Section 6.4: weights.decompose@1.0.0
Decomposes dense linear layers into low-rank SVDLinear factorizations.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from neuralops.adapters.registry import find_adapter
from neuralops.core.exceptions import IntegrationNotImplementedError
from neuralops.core.low_rank.svd_linear import SVDLinear
from neuralops.operations.base import Operation, OperationContract
from neuralops.operations.errors import PreconditionFailedError
from neuralops.operations.registry import register_operation

logger = logging.getLogger("NeuralOps.Operations.Weights.Decompose")

CONTRACT = OperationContract(
    name="weights.decompose",
    version="1.0.0",
    input_state="*",
    output_state="compressed",
    params_schema={
        "type": "object",
        "required": ["rank"],
        "properties": {
            "rank": {"type": "integer", "minimum": 1},
            "targets": {"type": "array", "items": {"type": "string"}},
            "target": {"type": "string"},
            "lora_rank": {"type": "integer", "default": 0},
            "lora_alpha": {"type": "number", "default": 32.0},
        },
        "additionalProperties": False,
    },
    preconditions=["linear_targets_exist"],
    postconditions=["all_targets_decomposed_to_rank"],
    declares_mutations=["*"],
    deterministic_given_seed=True,
    composable_with_self=True,
)


def _set_submodule(root: nn.Module, path: str, new_mod: nn.Module) -> None:
    """Replace submodule at dot-separated or index-separated path with new_mod."""
    parts = path.split(".")
    curr: Any = root
    for p in parts[:-1]:
        if p.isdigit():
            curr = curr[int(p)]
        else:
            curr = getattr(curr, p)
    last = parts[-1]
    if last.isdigit():
        curr[int(last)] = new_mod
    else:
        setattr(curr, last, new_mod)


def count_parameters(model: nn.Module) -> int:
    """Calculate the total number of parameters in a module."""
    return sum(p.numel() for p in model.parameters())


@register_operation(override=True)
class WeightsDecomposeOperation(Operation):
    """Operation for decomposing dense nn.Linear layers into SVDLinear representations."""

    def __init__(self) -> None:
        super().__init__(contract=CONTRACT)

    def apply(self, model: nn.Module, config: Dict[str, Any]) -> Tuple[nn.Module, Dict[str, Any]]:
        """Decompose target linear layers in model to rank-r SVDLinear layers.

        Args:
            model: The neural network module to decompose.
            config: Configuration dictionary specifying rank, target/targets, and optional lora params.

        Returns:
            Tuple of the modified model and an evidence dictionary.

        Raises:
            PreconditionFailedError: If no candidate linear layers are found.
            IntegrationNotImplementedError: If decomposition fails during execution.
        """
        rank = int(config.get("rank", 16))
        lora_rank = int(config.get("lora_rank", 0))
        lora_alpha = float(config.get("lora_alpha", 32.0))
        targets_list = config.get("targets")
        target_pattern = config.get("target")

        # Discover matching linear layers
        adapter = find_adapter(model)
        linear_targets: List[Tuple[str, nn.Linear]] = []

        if targets_list:
            for pattern in targets_list:
                for path, mod in adapter.resolve_target(model, pattern):
                    if isinstance(mod, nn.Linear) and not isinstance(mod, SVDLinear):
                        if (path, mod) not in linear_targets:
                            linear_targets.append((path, mod))
        elif target_pattern:
            for path, mod in adapter.resolve_target(model, target_pattern):
                if isinstance(mod, nn.Linear) and not isinstance(mod, SVDLinear):
                    linear_targets.append((path, mod))
        else:
            # Default: all nn.Linear modules in the model
            for name, mod in model.named_modules():
                if isinstance(mod, nn.Linear) and not isinstance(mod, SVDLinear):
                    linear_targets.append((name, mod))

        if not linear_targets:
            raise PreconditionFailedError(
                "weights.decompose failed precondition: no compressible nn.Linear modules found."
            )

        baseline_params = count_parameters(model)
        decomposed_records: List[Dict[str, Any]] = []

        for path, linear_mod in linear_targets:
            weight = linear_mod.weight.data
            bias = linear_mod.bias.data if linear_mod.bias is not None else None
            d_out, d_in = weight.shape
            r = max(1, min(rank, min(d_out, d_in)))

            svd_mod = SVDLinear(
                in_features=d_in,
                out_features=d_out,
                rank=r,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                bias=(bias is not None),
                dtype=weight.dtype,
                device=weight.device,
            )
            svd_mod.decompose_from_weight(weight=weight, bias=bias)

            _set_submodule(model, path, svd_mod)
            decomposed_records.append(
                {
                    "path": path,
                    "in_features": d_in,
                    "out_features": d_out,
                    "rank": r,
                    "lora_rank": lora_rank,
                }
            )

        final_params = count_parameters(model)
        evidence: Dict[str, Any] = {
            "operation": "weights.decompose",
            "version": self.version,
            "rank": rank,
            "modules_decomposed": len(decomposed_records),
            "baseline_total_parameters": baseline_params,
            "final_total_parameters": final_params,
            "compression_ratio": final_params / max(1, baseline_params),
            "details": decomposed_records,
        }

        logger.info(
            "Completed weights.decompose: %d modules decomposed. Parameters %d -> %d",
            len(decomposed_records),
            baseline_params,
            final_params,
        )

        return model, evidence


__all__ = ["WeightsDecomposeOperation", "CONTRACT"]
