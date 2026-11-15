#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Weights Prune Operation Contract.

Specification Section 6.6: weights.prune@1.0.0
Supports unstructured magnitude pruning, structured 2:4 sparsity, and channel pruning.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

from neuralops.adapters.registry import find_adapter
from neuralops.core.exceptions import IntegrationNotImplementedError
from neuralops.operations.base import Operation, OperationContract
from neuralops.operations.errors import PreconditionFailedError
from neuralops.operations.registry import register_operation

logger = logging.getLogger("NeuralOps.Operations.Weights.Prune")

CONTRACT = OperationContract(
    name="weights.prune",
    version="1.0.0",
    input_state="*",
    output_state="pruned",
    params_schema={
        "type": "object",
        "required": ["pattern"],
        "properties": {
            "sparsity": {"type": "number", "minimum": 0.05, "maximum": 0.95, "default": 0.5},
            "pattern": {"type": "string", "enum": ["unstructured", "2:4", "channel"]},
            "target": {"type": "string"},
            "targets": {"type": "array", "items": {"type": "string"}},
            "norm": {"type": "string", "enum": ["l1", "l2"], "default": "l2"},
        },
        "additionalProperties": False,
    },
    preconditions=["has_dense_parameters"],
    postconditions=["measured_sparsity >= sparsity"],
    declares_mutations=["*.weight"],
    deterministic_given_seed=True,
    composable_with_self=True,
)


def prune_unstructured(weight: torch.Tensor, sparsity: float) -> torch.Tensor:
    """Apply unstructured magnitude pruning to tensor."""
    if sparsity <= 0.0:
        return weight
    if sparsity >= 1.0:
        return torch.zeros_like(weight)
    k = int(sparsity * weight.numel())
    if k == 0:
        return weight
    flat = weight.abs().view(-1)
    threshold = torch.kthvalue(flat, k).values
    mask = weight.abs() > threshold
    return weight * mask.to(weight.dtype)


def prune_2_to_4(weight: torch.Tensor) -> torch.Tensor:
    """Apply NVIDIA Ampere-style structured 2:4 sparsity (2 zeros out of every 4 consecutive values)."""
    orig_shape = weight.shape
    # Reshape so that last dimension is a multiple of 4
    if orig_shape[-1] % 4 != 0:
        logger.warning(
            "Inner dimension %d is not divisible by 4; applying 50%% unstructured pruning fallback.",
            orig_shape[-1],
        )
        return prune_unstructured(weight, 0.5)

    flat = weight.view(-1, 4)
    # For each group of 4, zero out the two smallest values by absolute magnitude
    abs_vals = flat.abs()
    # topk largest values (2 largest)
    _, top_idx = torch.topk(abs_vals, k=2, dim=1, largest=True, sorted=False)
    mask = torch.zeros_like(flat, dtype=torch.bool)
    mask.scatter_(dim=1, index=top_idx, value=True)
    pruned_flat = flat * mask.to(flat.dtype)
    return pruned_flat.view(orig_shape)


def prune_channel(weight: torch.Tensor, sparsity: float, norm: str = "l2") -> torch.Tensor:
    """Apply structured channel (output row) pruning based on L1 or L2 norm."""
    if sparsity <= 0.0:
        return weight
    out_dim = weight.shape[0]
    num_to_prune = int(sparsity * out_dim)
    if num_to_prune == 0:
        return weight

    reshaped = weight.view(out_dim, -1)
    if norm.lower() == "l1":
        channel_norms = reshaped.abs().sum(dim=1)
    else:
        channel_norms = reshaped.norm(p=2, dim=1)

    threshold = torch.kthvalue(channel_norms, num_to_prune).values
    keep_mask = (channel_norms > threshold).unsqueeze(1)
    # Broadcast across remaining dimensions
    view_dims = [out_dim] + [1] * (weight.ndim - 1)
    keep_mask = keep_mask.view(*view_dims)
    return weight * keep_mask.to(weight.dtype)


@register_operation(override=True)
class WeightsPruneOperation(Operation):
    """Operation for weight pruning: unstructured, structured 2:4, and channel pruning."""

    def __init__(self) -> None:
        super().__init__(contract=CONTRACT)

    def apply(self, model: nn.Module, config: Dict[str, Any]) -> Tuple[nn.Module, Dict[str, Any]]:
        """Prune target layers in model according to specified pattern and sparsity.

        Args:
            model: Target neural network module.
            config: Pruning configuration containing pattern, sparsity, targets, norm, etc.

        Returns:
            Tuple of pruned model and evidence dictionary.

        Raises:
            PreconditionFailedError: If no candidate linear layers are found.
            IntegrationNotImplementedError: If the pruning pattern is invalid.
        """
        pattern = config.get("pattern", "unstructured")
        sparsity = float(config.get("sparsity", 0.5))
        norm_type = config.get("norm", "l2")
        targets_list = config.get("targets")
        target_pattern = config.get("target")

        adapter = find_adapter(model)
        linear_targets: List[Tuple[str, nn.Linear]] = []

        if targets_list:
            for pat in targets_list:
                for path, mod in adapter.resolve_target(model, pat):
                    if isinstance(mod, nn.Linear):
                        linear_targets.append((path, mod))
        elif target_pattern:
            for path, mod in adapter.resolve_target(model, target_pattern):
                if isinstance(mod, nn.Linear):
                    linear_targets.append((path, mod))
        else:
            for name, mod in model.named_modules():
                if isinstance(mod, nn.Linear):
                    linear_targets.append((name, mod))

        if not linear_targets:
            raise PreconditionFailedError(
                "weights.prune failed precondition: no dense nn.Linear parameters found."
            )

        pruned_records: List[Dict[str, Any]] = []
        total_elements = 0
        total_zeros = 0

        with torch.no_grad():
            for path, linear_mod in linear_targets:
                w = linear_mod.weight.data
                if pattern == "2:4":
                    w_pruned = prune_2_to_4(w)
                    target_sparsity = 0.5
                elif pattern == "unstructured":
                    w_pruned = prune_unstructured(w, sparsity=sparsity)
                    target_sparsity = sparsity
                elif pattern == "channel":
                    w_pruned = prune_channel(w, sparsity=sparsity, norm=norm_type)
                    target_sparsity = sparsity
                else:
                    raise IntegrationNotImplementedError(
                        f"weights.prune pattern '{pattern}' is not supported."
                    )

                linear_mod.weight.data.copy_(w_pruned)

                num_zeros = int((w_pruned == 0).sum().item())
                numel = w_pruned.numel()
                total_zeros += num_zeros
                total_elements += numel

                pruned_records.append(
                    {
                        "path": path,
                        "numel": numel,
                        "zeros": num_zeros,
                        "sparsity": num_zeros / max(1, numel),
                    }
                )

        measured_sparsity = total_zeros / max(1, total_elements)
        evidence: Dict[str, Any] = {
            "operation": "weights.prune",
            "version": self.version,
            "pattern": pattern,
            "requested_sparsity": sparsity,
            "measured_sparsity": measured_sparsity,
            "modules_pruned": len(pruned_records),
            "details": pruned_records,
        }

        logger.info(
            "Completed weights.prune (%s): %d modules pruned, measured sparsity: %.4f",
            pattern,
            len(pruned_records),
            measured_sparsity,
        )

        return model, evidence


__all__ = ["WeightsPruneOperation", "CONTRACT", "prune_unstructured", "prune_2_to_4", "prune_channel"]
