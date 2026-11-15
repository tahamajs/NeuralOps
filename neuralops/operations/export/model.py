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
Eliminates format stubs and supports all production targets:
hf, safetensors, vllm, sglang, onnx, gguf.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import torch.nn as nn

from neuralops.core.exceptions import ConfigError
from neuralops.operations.base import Operation, OperationContract
from neuralops.operations.registry import register_operation

logger = logging.getLogger("NeuralOps.Operations.Export")

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
            "output_dir": {"type": "string"},
            "model_name": {"type": "string"},
            "quantization": {"type": "string"},
            "max_shard_size": {"type": ["integer", "string"]},
        },
        "additionalProperties": True,
    },
    preconditions=["is_finite_model"],
    postconditions=["destination_directory_contains_valid_artifact"],
    declares_mutations=[],
    deterministic_given_seed=True,
    composable_with_self=False,
)


@register_operation
class ExportModelOperation(Operation):
    """Operation for exporting models across diverse formats and serving engines."""

    def __init__(self) -> None:
        """Initialize ExportModelOperation with contract."""
        super().__init__(contract=CONTRACT)

    def apply(self, model: nn.Module, config: Dict[str, Any]) -> Tuple[nn.Module, Dict[str, Any]]:
        """Apply model export across supported formats without stubs.

        Args:
            model: PyTorch model module to export.
            config: Export configuration dictionary specifying 'format', 'output_dir', etc.

        Returns:
            Tuple of (unmodified model, operation output metadata dictionary).

        Raises:
            ConfigurationError: If format is unsupported or configuration is invalid.
        """
        fmt = config.get("format", "hf").lower().strip()
        out_dir = Path(config.get("output_dir", "results/export"))
        out_dir.mkdir(parents=True, exist_ok=True)

        if fmt == "gguf":
            from neuralops.export.gguf import export_model_to_gguf

            out_path = export_model_to_gguf(
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

        elif fmt == "onnx":
            from neuralops.export.onnx import export_to_onnx

            sample_input = config.get("sample_input")
            is_recurrent = config.get("is_recurrent", False)
            out_path = export_to_onnx(
                model=model,
                output_path=out_dir / "model.onnx",
                sample_input=sample_input,
                is_recurrent=is_recurrent,
            )
            return model, {
                "format": "onnx",
                "output_path": str(out_path),
                "operation_version": self.version,
                "output_state": self.output_state,
            }

        elif fmt in ("vllm", "sglang"):
            from neuralops.export.serving import export_serving_bundle

            max_batch_size = config.get("max_batch_size", 32)
            max_sequence_length = config.get("max_sequence_length", 4096)
            out_path = export_serving_bundle(
                model=model,
                output_dir=out_dir,
                target_backend=fmt,
                max_batch_size=max_batch_size,
                max_sequence_length=max_sequence_length,
            )
            return model, {
                "format": fmt,
                "output_path": str(out_path),
                "target_backend": fmt,
                "operation_version": self.version,
                "output_state": self.output_state,
            }

        elif fmt in ("hf", "safetensors"):
            from neuralops.export.safetensors import save_sharded_safetensors

            max_shard_size = config.get("max_shard_size", 5 * 1024 * 1024 * 1024)
            res = save_sharded_safetensors(
                model_or_state_dict=model,
                output_dir=out_dir,
                max_shard_size=max_shard_size,
            )
            return model, {
                "format": fmt,
                "output_path": str(out_dir),
                "num_shards": res["num_shards"],
                "manifest_file": res["manifest_file"],
                "operation_version": self.version,
                "output_state": self.output_state,
            }

        else:
            raise ConfigError(
                f"Unsupported export format '{fmt}'. Choose from: 'hf', 'safetensors', 'vllm', 'sglang', 'onnx', 'gguf'."
            )


__all__ = [
    "CONTRACT",
    "ExportModelOperation",
]
