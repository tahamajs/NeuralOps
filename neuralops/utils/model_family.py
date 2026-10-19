#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Model Family Detection

Identifies model architectures, transformer families, and projection naming conventions.
"""

from __future__ import annotations

from neuralops.core.model_family import (
    CAUSAL_FAMILIES,
    FAMILIES,
    FAMILY_ALIASES,
    GPT_ABSOLUTE_POSITION_FAMILIES,
    ROTARY_FAMILIES,
    ModelAdapter,
    detect_family,
)

__all__ = [
    "CAUSAL_FAMILIES",
    "FAMILIES",
    "FAMILY_ALIASES",
    "GPT_ABSOLUTE_POSITION_FAMILIES",
    "ModelAdapter",
    "ROTARY_FAMILIES",
    "detect_family",
]
