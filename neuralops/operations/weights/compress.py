#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Weights Compress Operation Contract.

Specification Section 6.3: weights.compress@1.0.0
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch.nn as nn

from neuralops.core.errors import IntegrationNotImplementedError
from neuralops.operations.base import Operation, OperationContract
from neuralops.operations.registry import register_operation

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


@register_operation
class WeightsCompressOperation(Operation):
    def __init__(self):
        super().__init__(contract=CONTRACT)

    def apply(self, model: nn.Module, config: Dict[str, Any]) -> Tuple[nn.Module, Dict[str, Any]]:
        method = config.get("method", "aasvd")
        if method in ("aasvd", "svd", "asvd"):
            from neuralops.operations.weights.svd_aasvd import WeightsCompressAASVDOperation
            return WeightsCompressAASVDOperation().apply(model, config)
        raise IntegrationNotImplementedError(
            f"weights.compress method='{method}' implementation stub. "
            "Real AASVD algorithm is provided by neuralops.operations.weights.svd_aasvd."
        )
