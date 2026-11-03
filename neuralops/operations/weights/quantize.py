#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Weights Quantize Operation Contract.

Specification Section 6.5: weights.quantize@1.0.0
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch.nn as nn

from neuralops.core.errors import IntegrationNotImplementedError
from neuralops.operations.base import Operation, OperationContract
from neuralops.operations.registry import register_operation

CONTRACT = OperationContract(
    name="weights.quantize",
    version="1.0.0",
    input_state="*",
    output_state="quantized",
    params_schema={
        "type": "object",
        "required": ["bits", "method"],
        "properties": {
            "bits": {"type": "integer", "enum": [4, 8]},
            "method": {"type": "string", "enum": ["fp8", "int4_awq", "int4_gptq"]},
            "group_size": {"type": "integer", "default": 128},
        },
        "additionalProperties": False,
    },
    preconditions=["torch.cuda.is_available", "calibration_data_available"],
    postconditions=["all_linear_weights_quantized"],
    declares_mutations=["*.weight"],
    deterministic_given_seed=False,
    composable_with_self=True,
)


@register_operation
class WeightsQuantizeOperation(Operation):
    def __init__(self):
        super().__init__(contract=CONTRACT)

    def apply(self, model: nn.Module, config: Dict[str, Any]) -> Tuple[nn.Module, Dict[str, Any]]:
        method = config.get("method", "fp8")
        bits = config.get("bits", 8)
        if method == "fp8" or bits == 8:
            from neuralops.compression.quantization.fp8_compressor import FP8Compressor

            fp8_format = config.get("fp8_format", "e4m3")
            scale_type = config.get("scale_type", "per_tensor")
            compressor = FP8Compressor(model, scale_type=scale_type, fp8_format=fp8_format)
            compressed_model, stats = compressor.compress()
            evidence = {
                "operation": "weights.quantize",
                "method": method,
                "bits": bits,
                "layers_quantized": len(stats.get("layers", [])),
                "stats": stats,
            }
            return compressed_model, evidence
        elif method in ("int4_awq", "int4_gptq") or bits == 4:
            from neuralops.compression.quantization.awq import AWQCompressor

            group_size = config.get("group_size", 128)
            compressor = AWQCompressor(model, bits=4, group_size=group_size)
            compressed_model, stats = compressor.compress()
            evidence = {
                "operation": "weights.quantize",
                "method": method,
                "bits": 4,
                "group_size": group_size,
                "stats": stats,
            }
            return compressed_model, evidence

        raise IntegrationNotImplementedError(
            f"weights.quantize method '{method}' (bits={bits}) is not supported."
        )
