#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
FLA (Flash Linear Attention) Integration Subpackage.

Provides optional, fail-closed integration with upstream flash-linear-attention
while preserving all canonical repository invariants.
"""

from __future__ import annotations

import importlib.util

from neuralops.integrations.fla.adapter import SUPPORTED_MODEL_FAMILIES, FLALayerAdapter
from neuralops.integrations.fla.registry_bridge import (
    EXTENSION_ATTENTION_REGISTRY,
    SUPPORTED_FLA_ATTENTIONS,
    list_fla_attentions,
    register_fla_attention,
)


def fla_is_available() -> bool:
    """Check if the upstream 'fla' package is installed in the current environment."""
    return importlib.util.find_spec("fla") is not None


__all__ = [
    "fla_is_available",
    "FLALayerAdapter",
    "EXTENSION_ATTENTION_REGISTRY",
    "SUPPORTED_FLA_ATTENTIONS",
    "SUPPORTED_MODEL_FAMILIES",
    "list_fla_attentions",
    "register_fla_attention",
]
