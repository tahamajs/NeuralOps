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

import torch

from neuralops.core.exceptions import IntegrationNotImplementedError
from neuralops.integrations.fla.adapter import SUPPORTED_MODEL_FAMILIES, FLALayerAdapter
from neuralops.integrations.fla.registry_bridge import (
    EXTENSION_ATTENTION_REGISTRY,
    SUPPORTED_FLA_ATTENTIONS,
    list_fla_attentions,
    register_fla_attention,
)


def fla_is_available() -> bool:
    """Check if upstream 'fla' package is installed and CUDA is available (FLA is CUDA-only)."""
    if not torch.cuda.is_available():
        return False
    return importlib.util.find_spec("fla") is not None


def require_fla() -> None:
    """Raise IntegrationNotImplementedError with actionable guidance if FLA is unavailable."""
    if not torch.cuda.is_available():
        raise IntegrationNotImplementedError(
            "Flash Linear Attention (FLA) requires an NVIDIA CUDA accelerator, but CUDA is unavailable."
        )
    if importlib.util.find_spec("fla") is None:
        raise IntegrationNotImplementedError(
            "The 'fla' (Flash Linear Attention) package is required for this operation. "
            "Install it via: pip install 'neuralops[fla]' (requires Linux and CUDA GPU)."
        )


__all__ = [
    "fla_is_available",
    "require_fla",
    "FLALayerAdapter",
    "EXTENSION_ATTENTION_REGISTRY",
    "SUPPORTED_FLA_ATTENTIONS",
    "SUPPORTED_MODEL_FAMILIES",
    "list_fla_attentions",
    "register_fla_attention",
]
