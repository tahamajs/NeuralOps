#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unified Model Serialization & Export Interface

Provides a single export() entry point unifying safetensors, TorchScript,
ONNX, and serving runtime serialization formats.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

from neuralops.core.errors import IntegrationNotImplementedError

logger = logging.getLogger(__name__)

__all__ = ["export"]

SUPPORTED_FORMATS: tuple[str, ...] = ("safetensors", "torchscript", "onnx", "pt", "vllm", "gguf")


def export(
    model: nn.Module,
    format: str,
    output_path: str | Path,
    *,
    sample_input: Optional[torch.Tensor] = None,
    **kwargs: Any,
) -> str:
    """Export a neural network to the target serialization format.

    Args:
        model: Trained or compressed PyTorch module.
        format: Export target format ('safetensors', 'torchscript', 'onnx', 'pt', 'vllm').
        output_path: Destination file or directory path.
        sample_input: Example input tensor for trace-based exporters.
        **kwargs: Additional format-specific exporter options.

    Returns:
        String path to the saved artifact.
    """
    fmt = format.strip().lower()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if fmt not in SUPPORTED_FORMATS:
        raise IntegrationNotImplementedError(
            f"Export format '{format}' is not supported. "
            f"Supported formats: {', '.join(SUPPORTED_FORMATS)}."
        )

    if fmt == "safetensors":
        try:
            from safetensors.torch import save_file
            state_dict = model.state_dict()
            save_file(state_dict, str(out))
            return str(out)
        except ImportError:
            raise IntegrationNotImplementedError(
                "Export format 'safetensors' requires the 'safetensors' package: pip install safetensors"
            )

    elif fmt in ("pt", "torch"):
        torch.save(model.state_dict(), str(out))
        return str(out)

    elif fmt == "torchscript":
        if sample_input is not None:
            traced = torch.jit.trace(model, sample_input)
            traced.save(str(out))
        else:
            scripted = torch.jit.script(model)
            scripted.save(str(out))
        return str(out)

    elif fmt == "onnx":
        if sample_input is None:
            raise ValueError("ONNX export requires 'sample_input' tensor.")
        torch.onnx.export(
            model,
            sample_input,
            str(out),
            input_names=kwargs.get("input_names", ["input"]),
            output_names=kwargs.get("output_names", ["output"]),
            opset_version=kwargs.get("opset_version", 17),
        )
        return str(out)

    elif fmt == "gguf":
        from neuralops.export.gguf_exporter import export_to_gguf
        exported_path = export_to_gguf(
            model=model,
            output_path=out,
            architecture=kwargs.get("architecture", "llama"),
            model_name=kwargs.get("model_name", "model"),
            quantization=kwargs.get("quantization", "f16"),
            fuse_low_rank=kwargs.get("fuse_low_rank", False),
            metadata=kwargs.get("metadata", None),
        )
        return str(exported_path)

    elif fmt == "vllm":
        raise IntegrationNotImplementedError(
            "Native vLLM direct model engine compilation is scheduled for Roadmap v2.0.0. "
            "Please export to 'safetensors' format for vLLM HuggingFace model loading."
        )

    return str(out)
