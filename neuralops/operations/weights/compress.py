#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Weights Compress Operation Contract.

Specification Section 6.3: weights.compress@1.0.0
Supports SVD, AASVD, ASVD, and FWSVD low-rank compression hooks.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import torch.nn as nn

from neuralops.core.exceptions import IntegrationNotImplementedError
from neuralops.operations.base import Operation, OperationContract
from neuralops.operations.registry import register_operation

logger = logging.getLogger("NeuralOps.Operations.Weights.Compress")

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
    produces=["low_rank_projection"],
    ordering_constraints=["after:attention.replace"],
)


@register_operation(override=True)
class WeightsCompressOperation(Operation):
    """Operation for low-rank weight compression across diverse SVD variants."""

    def __init__(self) -> None:
        super().__init__(contract=CONTRACT)

    def apply(self, model: nn.Module, config: Dict[str, Any]) -> Tuple[nn.Module, Dict[str, Any]]:
        """Apply low-rank weight compression to model using the specified method.

        Args:
            model: Target neural network module.
            config: Compression configuration containing method, target_ratio, rank, etc.

        Returns:
            Tuple of the compressed model and an evidence dictionary.

        Raises:
            IntegrationNotImplementedError: If the method is unsupported.
        """
        method = config.get("method", "aasvd").lower()

        if method in ("aasvd", "svd"):
            from neuralops.operations.weights.svd_aasvd import WeightsCompressAASVDOperation

            return WeightsCompressAASVDOperation().apply(model, config)

        if method == "asvd":
            from neuralops.compression.svd.asvd_compressor import ASVDCompressor

            rank_param = config.get("rank")
            rank = int(rank_param) if isinstance(rank_param, int) else 32
            target_pattern = config.get("target")
            target_modules = [target_pattern] if target_pattern else None

            compressor = ASVDCompressor(
                model=model,
                rank=rank,
                fallback_to_original=False,
                target_attention_only=False,
                target_modules=target_modules,
            )
            compressed_model, stats = compressor.compress()
            evidence: Dict[str, Any] = {
                "operation": "weights.compress",
                "version": self.version,
                "method": "asvd",
                "stats": stats,
                "modules_compressed": stats.get("num_compressed", 0),
                "compression_ratio": stats.get("compression_ratio", 1.0),
            }
            return compressed_model, evidence

        if method == "fwsvd":
            from neuralops.compression.svd.fwsvd_compressor import FWSVDCompressor

            rank_param = config.get("rank")
            rank = int(rank_param) if isinstance(rank_param, int) else 32
            target_pattern = config.get("target")
            target_modules = [target_pattern] if target_pattern else None

            compressor = FWSVDCompressor(
                model=model,
                rank=rank,
                use_gradients=False,
                fallback_to_original=False,
                target_attention_only=False,
                target_modules=target_modules,
            )
            compressed_model, stats = compressor.compress()
            evidence = {
                "operation": "weights.compress",
                "version": self.version,
                "method": "fwsvd",
                "stats": stats,
                "modules_compressed": stats.get("compressed_layers", 0),
                "compression_ratio": stats.get("compression_ratio", 1.0),
            }
            return compressed_model, evidence

        raise IntegrationNotImplementedError(
            f"weights.compress method='{method}' is not implemented."
        )


__all__ = ["WeightsCompressOperation", "CONTRACT"]
