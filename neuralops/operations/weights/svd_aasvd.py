#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Activation-Aware SVD (AASVD) Weights Compression Operation.

Specification Section 6.3: weights.compress@1.0.0
Implementation for Phase 1 v0.1 Core.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from neuralops.adapters.registry import find_adapter
from neuralops.core.low_rank.svd_linear import SVDLinear
from neuralops.operations.base import Operation, OperationContract
from neuralops.operations.errors import OperationExecutionError, PreconditionFailedError
from neuralops.operations.registry import register_operation
from neuralops.operations.weights.svd.aasvd import (
    aasvd_decompose,
    allocate_rank_auto,
    compute_activation_covariance,
)

logger = logging.getLogger("NeuralOps.Operations.Weights.AASVD")

CONTRACT = OperationContract(
    name="weights.compress",
    version="1.0.0",
    input_state="*",
    output_state="compressed",
    params_schema={
        "type": "object",
        "required": ["method", "target_ratio"],
        "properties": {
            "method": {"type": "string", "enum": ["svd", "asvd", "fwsvd", "hnc_svd", "aasvd"]},
            "target": {"type": "string"},
            "target_ratio": {"type": "number", "minimum": 0.05, "maximum": 0.95},
            "rank": {"type": ["integer", "string"]},
            "rank_step": {"type": "integer", "default": 32},
            "whitening": {"type": "boolean", "default": False},
            "calibration_tokens": {"type": "integer", "default": 512},
        },
        "additionalProperties": False,
    },
    preconditions=["uncompressed_linear_weights_exist"],
    postconditions=["model.compression_ratio <= target_ratio"],
    declares_mutations=["*.mlp.*", "*.self_attn.*_proj"],
    deterministic_given_seed=True,
    composable_with_self=True,
)


def _set_submodule(root: nn.Module, path: str, new_mod: nn.Module) -> None:
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
    return sum(p.numel() for p in model.parameters())


@register_operation(override=True)
class WeightsCompressAASVDOperation(Operation):
    """Real Activation-Aware SVD (AASVD) weight compression."""

    def __init__(self) -> None:
        super().__init__(contract=CONTRACT)

    def apply(
        self,
        model: nn.Module,
        config: Dict[str, Any],
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        method = config.get("method", "aasvd")
        target_pattern = config.get("target", "*.mlp.*")
        target_ratio = float(config.get("target_ratio", 0.5))
        rank_param = config.get("rank", "auto")
        rank_step = int(config.get("rank_step", 32))
        whitening = bool(config.get("whitening", method == "aasvd"))
        calibration_tokens = int(config.get("calibration_tokens", 512))

        # 1. Discover adapter
        adapter = find_adapter(model)
        target_modules: List[Tuple[str, nn.Module]] = adapter.resolve_target(model, target_pattern)

        # Filter only nn.Linear that are not already SVDLinear
        linear_targets: List[Tuple[str, nn.Linear]] = [
            (path, mod)
            for path, mod in target_modules
            if isinstance(mod, nn.Linear) and not isinstance(mod, SVDLinear)
        ]

        if not linear_targets:
            raise PreconditionFailedError(
                f"weights.compress failed precondition: no compressible nn.Linear modules "
                f"found matching target '{target_pattern}'. Discovered {len(target_modules)} candidates."
            )

        baseline_params = count_parameters(model)
        original_target_params = sum(count_parameters(mod) for _, mod in linear_targets)

        logger.info(
            "Found %d candidate linear layers matching target '%s' for AASVD compression.",
            len(linear_targets),
            target_pattern,
        )

        # 2. Calibration covariance (if whitening enabled or method is aasvd)
        cov_matrices: Dict[str, torch.Tensor] = {}
        if whitening or method == "aasvd":
            logger.info("Computing activation covariance over %d tokens...", calibration_tokens)
            try:
                cov_matrices = compute_activation_covariance(
                    model=model,
                    target_modules=[(p, m) for p, m in linear_targets],
                    calibration_tokens=calibration_tokens,
                )
            except Exception as e:
                logger.warning("Activation covariance collection warning: %s; falling back to unweighted SVD", e)
                whitening = False

        # 3. Decompose and replace
        compressed_records = []
        compressed_target_params = 0

        for path, linear_mod in linear_targets:
            weight = linear_mod.weight.data
            bias = linear_mod.bias.data if linear_mod.bias is not None else None
            d_out, d_in = weight.shape
            max_rank = min(d_out, d_in)

            # Determine rank
            if rank_param == "auto" or rank_param is None:
                # Target parameter budget: (d_in * r + r * d_out) ~= target_ratio * (d_in * d_out)
                ideal_r = int(round((target_ratio * d_in * d_out) / (d_in + d_out)))
                ideal_r = max(4, min(ideal_r, max_rank))
                if ideal_r > rank_step:
                    r = (ideal_r // rank_step) * rank_step
                else:
                    r = ideal_r
                r = max(4, min(r, max_rank))
            else:
                try:
                    r = int(rank_param)
                except (ValueError, TypeError):
                    r = max(4, int(max_rank * target_ratio))
                r = max(1, min(r, max_rank))

            # Run AASVD decomposition
            cov = cov_matrices.get(path)
            U_r, S_r, V_r, energy_retention = aasvd_decompose(
                weight=weight,
                activation_cov=cov,
                rank=r,
                whitening=whitening,
            )

            # Instantiate SVDLinear
            svd_mod = SVDLinear(
                in_features=d_in,
                out_features=d_out,
                rank=r,
                lora_rank=0,
                bias=(bias is not None),
                dtype=weight.dtype,
                device=weight.device,
            )

            with torch.no_grad():
                svd_mod.U.data.copy_(U_r.to(svd_mod.U.device, svd_mod.U.dtype))
                svd_mod.S.data.copy_(S_r.to(svd_mod.S.device, svd_mod.S.dtype))
                svd_mod.V.data.copy_(V_r.t().to(svd_mod.V.device, svd_mod.V.dtype))
                if bias is not None and svd_mod.bias is not None:
                    svd_mod.bias.data.copy_(bias.to(svd_mod.bias.device, svd_mod.bias.dtype))

            # Replace in model
            _set_submodule(model, path, svd_mod)
            mod_params = count_parameters(svd_mod)
            compressed_target_params += mod_params

            compressed_records.append({
                "path": path,
                "in_features": d_in,
                "out_features": d_out,
                "rank": r,
                "energy_retention": energy_retention,
                "original_parameters": count_parameters(linear_mod),
                "compressed_parameters": mod_params,
            })

        final_total_params = count_parameters(model)
        compression_ratio = final_total_params / baseline_params

        evidence = {
            "operation": "weights.compress",
            "version": "1.0.0",
            "method": method,
            "target_pattern": target_pattern,
            "target_ratio": target_ratio,
            "whitening": whitening,
            "calibration_tokens": calibration_tokens,
            "modules_compressed": len(compressed_records),
            "baseline_total_parameters": baseline_params,
            "final_total_parameters": final_total_params,
            "original_target_parameters": original_target_params,
            "compressed_target_parameters": compressed_target_params,
            "compression_ratio": compression_ratio,
            "details": compressed_records,
        }

        logger.info(
            "Completed AASVD compression: %d modules compressed. Parameters %d -> %d (ratio: %.4f)",
            len(compressed_records),
            baseline_params,
            final_total_params,
            compression_ratio,
        )

        return model, evidence


__all__ = ["WeightsCompressAASVDOperation", "CONTRACT"]
