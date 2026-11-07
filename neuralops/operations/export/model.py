#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Export Model Operation Contract.

Specification Section 6.13: export.model@1.0.0
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import torch.nn as nn

from neuralops.core.exceptions import IntegrationNotImplementedError
from neuralops.operations.base import Operation, OperationContract
from neuralops.operations.registry import register_operation

CONTRACT = OperationContract(
    name="export.model",
    version="1.0.0",
    input_state="*",
    output_state="ServingArtifact[engine]",
    params_schema={
        "type": "object",
        "required": ["format"],
        "properties": {
            "format": {"type": "string", "enum": ["hf", "safetensors", "vllm", "sglang", "onnx", "gguf"]},
            "verify_reload": {"type": "boolean", "default": True},
        },
        "additionalProperties": False,
    },
    preconditions=["is_finite_model"],
    postconditions=["destination_directory_contains_valid_artifact"],
    declares_mutations=[],
    deterministic_given_seed=True,
    composable_with_self=False,
)


@register_operation
class ExportModelOperation(Operation):
    def __init__(self):
        super().__init__(contract=CONTRACT)

    def apply(self, model: nn.Module, config: Dict[str, Any]) -> Tuple[nn.Module, Dict[str, Any]]:
        fmt = config.get("format", "hf")
        if fmt == "gguf":
            from neuralops.export.gguf_exporter import export_to_gguf

            out_dir = Path(config.get("output_dir", "."))
            out_path = export_to_gguf(
                model=model,
                output_path=out_dir,
                model_name=config.get("model_name", "model"),
                quantization=config.get("quantization", "f16"),
            )
            return model, {
                "format": "gguf",
                "output_path": str(out_path),
                "operation_version": self.version,
                "output_state": self.output_state,
            }
        elif fmt not in ("hf", "safetensors"):
            raise IntegrationNotImplementedError(f"export.model format='{fmt}' requires Serving Engine integration.")
        return model, {"format": fmt, "operation_version": self.version, "output_state": self.output_state}
