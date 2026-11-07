#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
MLP Shrinkage Surgery: Intermediate Dimension Reduction for SwiGLU and GELU MLPs.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from neuralops.surgery.registry import register_surgery

logger = logging.getLogger(__name__)


def _shrink_linear_out(layer: nn.Linear, target_out: int) -> nn.Linear:
    """Shrink output features of nn.Linear while preserving top weights."""
    in_features = layer.in_features
    bias = layer.bias is not None
    device = layer.weight.device
    dtype = layer.weight.dtype

    new_layer = nn.Linear(in_features, target_out, bias=bias, device=device, dtype=dtype)
    with torch.no_grad():
        new_layer.weight.copy_(layer.weight[:target_out, :])
        if bias and layer.bias is not None:
            new_layer.bias.copy_(layer.bias[:target_out])
    return new_layer


def _shrink_linear_in(layer: nn.Linear, target_in: int) -> nn.Linear:
    """Shrink input features of nn.Linear while preserving top weights."""
    out_features = layer.out_features
    bias = layer.bias is not None
    device = layer.weight.device
    dtype = layer.weight.dtype

    new_layer = nn.Linear(target_in, out_features, bias=bias, device=device, dtype=dtype)
    with torch.no_grad():
        new_layer.weight.copy_(layer.weight[:, :target_in])
        if bias and layer.bias is not None:
            new_layer.bias.copy_(layer.bias)
    return new_layer


def _shrink_swiglu_mlp(mlp: nn.Module, ratio: float) -> nn.Module:
    """Shrink intermediate dimension for SwiGLU MLP (gate_proj, up_proj, down_proj)."""
    gate = getattr(mlp, "gate_proj", None)
    up = getattr(mlp, "up_proj", None)
    down = getattr(mlp, "down_proj", None)

    if gate is None or up is None or down is None:
        return mlp

    curr_inter = gate.out_features
    target_inter = max(16, int(curr_inter * ratio))

    if target_inter >= curr_inter:
        return mlp

    mlp.gate_proj = _shrink_linear_out(gate, target_inter)
    mlp.up_proj = _shrink_linear_out(up, target_inter)
    mlp.down_proj = _shrink_linear_in(down, target_inter)

    return mlp


def _shrink_gelu_mlp(mlp: nn.Module, ratio: float) -> nn.Module:
    """Shrink intermediate dimension for 2-layer GELU MLP (c_fc/c_proj, fc1/fc2, lin1/lin2)."""
    fc1 = getattr(mlp, "c_fc", getattr(mlp, "fc1", getattr(mlp, "linear1", getattr(mlp, "lin1", None))))
    fc2 = getattr(mlp, "c_proj", getattr(mlp, "fc2", getattr(mlp, "linear2", getattr(mlp, "lin2", None))))

    if fc1 is None or fc2 is None:
        return mlp

    # Check for HuggingFace Conv1D (used in GPT-2)
    if hasattr(fc1, "nx") and hasattr(fc1, "nf") and hasattr(fc2, "nx") and hasattr(fc2, "nf"):
        curr_inter = fc1.nf
        target_inter = max(16, int(curr_inter * ratio))
        if target_inter >= curr_inter:
            return mlp

        new_fc1 = type(fc1)(target_inter, fc1.nx)
        new_fc1.weight.data.copy_(fc1.weight.data[:, :target_inter])
        if fc1.bias is not None and new_fc1.bias is not None:
            new_fc1.bias.data.copy_(fc1.bias.data[:target_inter])

        new_fc2 = type(fc2)(fc2.nf, target_inter)
        new_fc2.weight.data.copy_(fc2.weight.data[:target_inter, :])
        if fc2.bias is not None and new_fc2.bias is not None:
            new_fc2.bias.data.copy_(fc2.bias.data)

        if hasattr(mlp, "c_fc"):
            mlp.c_fc = new_fc1
            mlp.c_proj = new_fc2
        return mlp

    if not isinstance(fc1, nn.Linear) or not isinstance(fc2, nn.Linear):
        return mlp

    curr_inter = fc1.out_features
    target_inter = max(16, int(curr_inter * ratio))

    if target_inter >= curr_inter:
        return mlp

    if hasattr(mlp, "c_fc"):
        mlp.c_fc = _shrink_linear_out(fc1, target_inter)
        mlp.c_proj = _shrink_linear_in(fc2, target_inter)
    elif hasattr(mlp, "fc1"):
        mlp.fc1 = _shrink_linear_out(fc1, target_inter)
        mlp.fc2 = _shrink_linear_in(fc2, target_inter)
    elif hasattr(mlp, "linear1"):
        mlp.linear1 = _shrink_linear_out(fc1, target_inter)
        mlp.linear2 = _shrink_linear_in(fc2, target_inter)
    elif hasattr(mlp, "lin1"):
        mlp.lin1 = _shrink_linear_out(fc1, target_inter)
        mlp.lin2 = _shrink_linear_in(fc2, target_inter)

    return mlp


def _get_layer_idx(name: str) -> int | None:
    """Extract layer index from module path if present."""
    for part in name.split("."):
        if part.isdigit():
            return int(part)
    return None


@register_surgery(
    name="shrink_mlp",
    category="mlp",
    compat=["llama", "qwen", "gpt2", "mistral", "phi"],
    requires_gpu=False,
    requires_calibration=False,
    reversibility="low",
    cost_estimate="medium",
    _experimental=True,
)
def shrink_mlp(
    model: Any,
    ratio: float = 0.75,
    layers: str | list[int] = "all",
    **kwargs: Any,
) -> Any:
    """Reduce intermediate hidden dimension of MLP blocks across Transformer layers."""
    if not (0.0 < ratio <= 1.0):
        raise ValueError(f"Shrinkage ratio must be in (0.0, 1.0], got {ratio}")

    if not isinstance(model, nn.Module):
        return model

    target_layers: set[int] | None = None
    if layers != "all" and isinstance(layers, (list, tuple, set)):
        target_layers = set(layers)

    shrunk_count = 0
    for name, module in model.named_modules():
        mod_name_lower = name.lower()
        if "mlp" in mod_name_lower or "feedforward" in mod_name_lower or "ffn" in mod_name_lower:
            if target_layers is not None:
                layer_idx = _get_layer_idx(name)
                if layer_idx is not None and layer_idx not in target_layers:
                    continue

            # Check for SwiGLU
            if hasattr(module, "gate_proj") and hasattr(module, "up_proj") and hasattr(module, "down_proj"):
                _shrink_swiglu_mlp(module, ratio)
                shrunk_count += 1
            # Check for GELU / 2-layer
            elif (
                hasattr(module, "c_fc")
                or hasattr(module, "fc1")
                or hasattr(module, "linear1")
                or hasattr(module, "lin1")
            ):
                _shrink_gelu_mlp(module, ratio)
                shrunk_count += 1

    logger.info("Shrunk intermediate dimension on %d MLP blocks with ratio %.2f", shrunk_count, ratio)
    return model


__all__ = ["shrink_mlp"]
