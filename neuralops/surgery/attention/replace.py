#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Attention Surgery Operator wrapping NeuralOps attention replacement engines.
"""

from __future__ import annotations

from typing import Any

from neuralops.api import replace_attention as _core_replace_attention
from neuralops.surgery.registry import register_surgery


@register_surgery(
    name="replace_attention",
    category="attention",
    compat=["llama", "qwen", "gpt2", "mistral", "gemma", "phi"],
    requires_gpu=False,
    requires_calibration=False,
    reversibility="none",
    cost_estimate="low",
)
def replace_attention(
    model: Any,
    attention: str = "chunk_gla",
    layers: str | list[int] = "all",
    **kwargs: Any,
) -> Any:
    """Replace attention modules in target model with subquadratic attention variants."""
    target_name = attention
    if target_name in ("chunk_gla", "gla"):
        target_name = "gla_attention"
    target_layers = None if layers == "all" else (layers if isinstance(layers, list) else None)
    return _core_replace_attention(model, attention=target_name, layers=target_layers, **kwargs)


__all__ = ["replace_attention"]
