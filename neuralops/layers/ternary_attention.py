#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Ternary {-1, 0, 1} Attention Layer Facade."""

from __future__ import annotations

from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.layers.bitnet_attention import (
    BitLinear,
    BitNetAttention,
    activation_quant_ste,
    weight_quant_ste,
)

TernaryAttention = BitNetAttention
ATTENTION_REGISTRY.register("ternary_attention", is_alias=True)(TernaryAttention)

__all__ = [
    "TernaryAttention",
    "BitNetAttention",
    "BitLinear",
    "weight_quant_ste",
    "activation_quant_ste",
]
