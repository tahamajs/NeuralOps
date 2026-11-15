#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
GGUF Model Exporter.

Exposes llama.cpp compatible GGUF v3 serialization pipelines, quantization encoders,
and writer abstractions for edge deployment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch.nn as nn

from neuralops.export.gguf_exporter import (
    GGUF_DEFAULT_ALIGNMENT,
    GGUF_MAGIC,
    GGUF_VERSION,
    GGMLType,
    GGUFValueType,
    GGUFWriter,
    export_to_gguf,
    quantize_to_q8_0,
)


def export_model_to_gguf(
    model: nn.Module,
    output_path: Union[str, Path],
    architecture: str = "llama",
    model_name: str = "neuralops_model",
    context_length: int = 2048,
    embedding_length: Optional[int] = None,
    quantization: str = "f16",
    fuse_low_rank: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Export a PyTorch neural network to llama.cpp compatible GGUF v3 container format.

    Args:
        model: PyTorch model module to export.
        output_path: Destination file or directory path for .gguf file.
        architecture: GGUF architecture string (e.g. 'llama', 'qwen2', 'gpt2').
        model_name: Name identifier of the model.
        context_length: Maximum context sequence length.
        embedding_length: Hidden dimension or embedding length (inferred if None).
        quantization: Target GGML quantization format ('f32', 'f16', 'q8_0').
        fuse_low_rank: Whether to algebraically fuse low-rank decomposed weights.
        metadata: Optional custom key-value metadata to write into GGUF header.

    Returns:
        Path to the exported GGUF artifact.
    """
    out_p = Path(output_path)
    if out_p.is_dir() or not out_p.suffix:
        out_p.mkdir(parents=True, exist_ok=True)
        target_path = out_p / f"{model_name}.gguf"
    else:
        out_p.parent.mkdir(parents=True, exist_ok=True)
        target_path = out_p

    return export_to_gguf(
        model=model,
        output_path=target_path,
        architecture=architecture,
        model_name=model_name,
        context_length=context_length,
        embedding_length=embedding_length,
        quantization=quantization,
        fuse_low_rank=fuse_low_rank,
        metadata=metadata,
    )


__all__ = [
    "GGMLType",
    "GGUFValueType",
    "GGUFWriter",
    "GGUF_DEFAULT_ALIGNMENT",
    "GGUF_MAGIC",
    "GGUF_VERSION",
    "export_model_to_gguf",
    "export_to_gguf",
    "quantize_to_q8_0",
]
