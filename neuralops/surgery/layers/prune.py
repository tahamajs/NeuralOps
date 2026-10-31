#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Layer Pruning Surgery: Redundant Layer Shedding and Depth Truncation.
"""

from __future__ import annotations

import logging
from typing import Any

import torch.nn as nn

from neuralops.surgery.registry import register_surgery

logger = logging.getLogger(__name__)


def _find_decoder_layers(model: nn.Module) -> tuple[nn.Module, str, nn.ModuleList] | None:
    """Locate the primary Transformer decoder ModuleList in model."""
    for parent_name, parent_mod in model.named_modules():
        for attr in ("layers", "h", "block", "blocks"):
            if hasattr(parent_mod, attr):
                candidate = getattr(parent_mod, attr)
                if isinstance(candidate, nn.ModuleList) and len(candidate) > 1:
                    return parent_mod, attr, candidate
    return None


@register_surgery(
    name="prune_layers",
    category="layers",
    compat=["llama", "qwen", "gpt2", "mistral", "phi"],
    requires_gpu=False,
    requires_calibration=False,
    reversibility="low",
    cost_estimate="low",
    _experimental=True,
)
def prune_layers(
    model: Any,
    keep_layers: list[int] | None = None,
    drop_count: int | None = None,
    **kwargs: Any,
) -> Any:
    """Shed or prune specific Transformer decoder blocks from model backbone."""
    if not isinstance(model, nn.Module):
        return model

    info = _find_decoder_layers(model)
    if info is None:
        logger.warning("Could not identify Transformer layers ModuleList to prune.")
        return model

    parent, attr_name, layers = info
    total = len(layers)

    if keep_layers is not None:
        selected_indices = [idx for idx in keep_layers if 0 <= idx < total]
    elif drop_count is not None and drop_count > 0:
        # Drop drop_count layers from the late-middle blocks
        num_keep = max(1, total - drop_count)
        selected_indices = list(range(num_keep))
    else:
        # Default: keep all
        return model

    if not selected_indices:
        raise ValueError("Cannot prune all layers; at least 1 layer must be preserved.")

    new_list = nn.ModuleList([layers[i] for i in selected_indices])
    setattr(parent, attr_name, new_list)

    # Update config if present
    if hasattr(model, "config") and hasattr(model.config, "num_hidden_layers"):
        model.config.num_hidden_layers = len(new_list)

    logger.info("Pruned layers from %d to %d (kept indices: %s)", total, len(new_list), selected_indices)
    return model


__all__ = ["prune_layers"]
