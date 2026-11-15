#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""SambaMoE (Samba Mixture of Experts) architecture layer facade."""

from __future__ import annotations

from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.layers.ssm.samba_moe import MinimalMoEFallback, SambaMoEBlock

SambaMoE = SambaMoEBlock
ATTENTION_REGISTRY.register("sambamoe", is_alias=True)(SambaMoE)

__all__ = ["SambaMoEBlock", "SambaMoE", "MinimalMoEFallback"]
