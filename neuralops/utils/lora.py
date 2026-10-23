#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
LoRA Layer Utilities

Helper utilities for low-rank adaptation matrix creation, merging, and decomposition.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable

import torch
import torch.nn as nn
from torch.nn.init import kaiming_uniform_, zeros_

logger = logging.getLogger(__name__)


from neuralops.core.peft.lora import LoRALinear


DEFAULT_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "q_lin",
    "k_lin",
    "v_lin",
    "out_lin",
    "query",
    "key",
    "value",
    "dense",
    "q",
    "k",
    "v",
    "o",
    "wi",
    "wo",
    "wi_0",
    "wi_1",
    "c_attn",
    "c_proj",
    "c_fc",
    "in_proj",
    "out_proj",
    "down",
    "up",
    "fc1",
    "fc2",
    "Wqkv",
    "qkv_proj",
    "qkv",
)


def find_all_linear_submodule_names(model: nn.Module) -> set[str]:
    """Dynamically extract all Linear/Conv1D layer leaf names from model, excluding task heads."""
    names = set()
    for name, module in model.named_modules():
        is_linear_like = isinstance(
            module, (nn.Linear, nn.Conv1d)
        ) or module.__class__.__name__ in ("Conv1D", "Linear", "LoRALinear")
        if is_linear_like:
            if not any(head in name.lower() for head in ("score", "classifier", "lm_head")):
                leaf_name = name.split(".")[-1]
                names.add(leaf_name)
    return names


def apply_lora(
    model: nn.Module,
    r: int = 16,
    alpha: float = 32.0,
    dropout: float = 0.05,
    target_modules: Iterable[str] | str | None = None,
) -> tuple[nn.Module, int]:
    """Freeze base model parameters and inject LoRALinear into matching target modules.

    Args:
        model: PyTorch module (e.g. Qwen2, DistilBERT, RoBERTa, GPT-2).
        r: LoRA rank dimension.
        alpha: LoRA scaling factor (scaling = alpha / r).
        dropout: Dropout probability applied to LoRA adapter input.
        target_modules: Iterable or comma-separated string of target submodule names.

    Returns:
        Tuple of (model, number_of_adapted_linear_layers).
    """
    if target_modules is None:
        targets = set(DEFAULT_TARGET_MODULES)
    elif isinstance(target_modules, str):
        targets = {t.strip() for t in target_modules.split(",") if t.strip()}
    else:
        targets = set(target_modules)

    for p in model.parameters():
        p.requires_grad = False

    adapted_count = 0

    def _replace_linear_recursive(module: nn.Module, prefix: str = "") -> None:
        nonlocal adapted_count
        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            is_target = any(target in name for target in targets) or any(
                target == name for target in targets
            )
            is_linear_like = isinstance(child, nn.Linear) or child.__class__.__name__ == "Conv1D"

            if is_linear_like and is_target:
                if any(head in name.lower() for head in ("score", "classifier", "lm_head")):
                    continue
                wrapped = LoRALinear(child, r=r, alpha=alpha, dropout=dropout)
                setattr(module, name, wrapped)
                adapted_count += 1
            else:
                _replace_linear_recursive(child, full_name)

    _replace_linear_recursive(model)
    return model, adapted_count


def get_lora_trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Return all parameters that require gradients (LoRA adapters + task heads)."""
    return [p for p in model.parameters() if p.requires_grad]


def save_lora_weights(model: nn.Module, save_path: str, metadata: dict | None = None) -> None:
    """Save only trainable parameters (LoRA weights + task head) to disk."""
    trainable_state = {
        name: param.detach().cpu()
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    payload = {
        "state_dict": trainable_state,
        "metadata": metadata or {},
    }
    with open(save_path, "wb") as f:
        torch.save(payload, f)


def load_lora_weights(model: nn.Module, load_path: str, strict: bool = False) -> dict:
    """Load trainable parameters from saved checkpoint into model."""
    with open(load_path, "rb") as f:
        payload = torch.load(f, map_location="cpu", weights_only=False)
    state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    return {
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "metadata": payload.get("metadata", {}) if isinstance(payload, dict) else {},
    }

def make_lora_linear(
    in_features: int,
    out_features: int,
    bias: bool = True,
    lora_rank: int = 16,
    lora_alpha: float = 32.0,
    **kwargs,
) -> LoRALinear:
    """Factory for creating LoRALinear from dimensions (int-based API)."""
    base = nn.Linear(in_features, out_features, bias=bias)
    return LoRALinear(base, r=lora_rank, alpha=lora_alpha, dropout=0.0)
