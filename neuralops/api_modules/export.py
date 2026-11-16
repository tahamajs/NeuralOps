# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Model export, hub push, load, and AOT compilation: export_model, export, push_to_hub, load_model, compile_to_so.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn

from neuralops.export.exporter import export_model as _core_export_model
from neuralops.export.exporter import load_model as _core_load_model

logger = logging.getLogger("NeuralOps.API")


def export_model(
    model: nn.Module,
    output_dir: Union[str, Path],
    format: str = "safetensors",
    validation_level: Optional[str] = None,
    **kwargs: Any,
) -> Path:
    """Export a NeuralOps-modified or compressed model to a production artifact format.

    Persists `model._neuralops_metadata` into `neuralops_config.json` alongside weights.

    Args:
        model: The PyTorch/HuggingFace model to export.
        output_dir: Destination folder path.
        format: Export format ('safetensors', 'onnx', or 'torchscript').
        **kwargs: Additional exporter arguments.

    Returns:
        Path to output directory or exported artifact file.
    """
    logger.info("Exporting model to format '%s' at %s", format, output_dir)
    return _core_export_model(model=model, output_dir=output_dir, format=format, **kwargs)


# High-Level Public Ergonomics Alias
export = export_model


def push_to_hub(
    model: nn.Module,
    repo_id: str,
    tokenizer: Any = None,
    token: Optional[str] = None,
    private: bool = False,
    commit_message: str = "Upload NeuralOps model",
    **kwargs: Any,
) -> str:
    """Export a self-contained NeuralOps model and push directly to Hugging Face Hub.

    Configures auto_map so the model can be loaded by anyone via:
        AutoModel.from_pretrained(repo_id, trust_remote_code=True)

    Args:
        model: PyTorch/HuggingFace model to upload.
        repo_id: Target HuggingFace repository ID (e.g., 'username/my-model').
        tokenizer: Optional tokenizer to upload alongside model.
        token: HuggingFace API token (or read from HF_TOKEN env var).
        private: Whether the repository should be private.
        commit_message: Git commit message for upload.

    Returns:
        URL string of the uploaded HuggingFace Hub repository.
    """
    from neuralops.models.hub import push_model_to_hub as _push

    return _push(
        model=model,
        repo_id=repo_id,
        tokenizer=tokenizer,
        token=token,
        private=private,
        commit_message=commit_message,
    )


def load_model(
    export_dir: Union[str, Path],
    base_model: Optional[nn.Module] = None,
) -> nn.Module:
    """Load an exported NeuralOps model and replay surgery metadata.

    Reads `neuralops_config.json` and replays all recorded `_neuralops_metadata`
    operations sequentially before loading weights.

    Args:
        export_dir: Directory containing `neuralops_config.json` and weight files.
        base_model: Optional pre-instantiated base model.

    Returns:
        The reconstructed neuralops model.
    """
    logger.info("Loading exported model from '%s'", export_dir)
    return _core_load_model(export_dir=export_dir, base_model=base_model)


def compile_to_so(
    model: nn.Module,
    sample_inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    output_path: Union[str, Path] = "model_aot.so",
    dynamic_batch: bool = True,
    dynamic_seq_len: bool = True,
    min_seq_len: int = 1,
    max_seq_len: int = 32768,
    device: Optional[Union[str, torch.device]] = None,
) -> Any:
    """Compile a model to a standalone C++ AOTInductor shared library (.so or .pt2)."""
    from neuralops.export.exporter import compile_to_so as _compile_to_so

    return _compile_to_so(
        model=model,
        sample_inputs=sample_inputs,
        output_path=output_path,
        dynamic_batch=dynamic_batch,
        dynamic_seq_len=dynamic_seq_len,
        min_seq_len=min_seq_len,
        max_seq_len=max_seq_len,
        device=device,
    )
