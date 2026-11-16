# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Compression API: compress, compress_model.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch.nn as nn

logger = logging.getLogger("NeuralOps.API")


def compress(
    model: nn.Module,
    method: str = "svd",
    rank: int = 32,
    layers: Optional[List[int]] = None,
    validation_level: Optional[str] = None,
    **kwargs: Any,
) -> nn.Module:
    # Rule: no function in api.py or core/compression_pipeline.py mutates an input dict or model class;
    # if it needs to change one, it copies first. Stats travel as an explicit attribute.
    """Compress model weights using SVD, low-rank factorization, or pruning.

    Records the operation in `model._neuralops_metadata`.

    Args:
        model: PyTorch or HuggingFace model to compress.
        method: Compression engine name from `neuralops.list_compressors()`.
        rank: Target rank for low-rank decomposition (default: 32).
        layers: Optional list of layer indices to compress (default: None compresses all).
        validation_level: Optional validation level ('strict', 'lenient', or None) for checking compression bounds.
        **kwargs: Extra parameters passed to the compressor.

    Returns:
        The compressed model.
    """
    from neuralops.core.compression_pipeline import compress_model as _core_compress_model

    logger.info("Starting model compression '%s' (rank=%s)", method, rank)
    comp_step: Dict[str, Any] = {
        "method": method,
        "rank": rank,
        **kwargs,
    }
    if layers is not None:
        comp_step["layer_indices"] = layers

    config = {
        "compressors": [comp_step],
    }

    out_model = _core_compress_model(model=model, config=config)
    logger.info("Model compression completed successfully: %s", method)

    meta_entry: Dict[str, Any] = {
        "op": "compress",
        "type": method,
        "kwargs": {
            "rank": rank,
            "layers": layers,
            **kwargs,
        },
    }
    if not hasattr(out_model, "_neuralops_metadata") or out_model._neuralops_metadata is None:
        out_model._neuralops_metadata = []
    out_model._neuralops_metadata.append(meta_entry)

    if hasattr(out_model, "config") and hasattr(out_model.config, "to_dict"):
        if not hasattr(out_model.config, "neuralops_history") or out_model.config.neuralops_history is None:
            out_model.config.neuralops_history = []
        out_model.config.neuralops_history.append(meta_entry)

    # Attach stats directly to the model as an attribute; do not mutate model class or __iter__
    stats = getattr(out_model, "_compression_stats", {})
    out_model._neuralops_compression_stats = stats

    return out_model


# Canonical compression pipeline alias
compress_model = compress
