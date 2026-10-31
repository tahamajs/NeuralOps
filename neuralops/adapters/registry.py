#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Architecture Adapter Registry.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch.nn as nn

from neuralops.adapters.base import ArchitectureAdapter

logger = logging.getLogger("NeuralOps.Adapters.Registry")

_ADAPTER_REGISTRY: List[ArchitectureAdapter] = []


def register_adapter(adapter: ArchitectureAdapter) -> ArchitectureAdapter:
    """Register an ArchitectureAdapter instance into the global registry."""
    if adapter not in _ADAPTER_REGISTRY:
        _ADAPTER_REGISTRY.append(adapter)
        logger.debug(f"Registered adapter for family '{adapter.family}'")
    return adapter


def find_adapter(model: nn.Module) -> Optional[ArchitectureAdapter]:
    """Find the first registered adapter that can handle the model, or None if unsupported."""
    for adapter in _ADAPTER_REGISTRY:
        if adapter.can_handle(model):
            return adapter
    return None


def list_adapters() -> List[str]:
    """Return list of all registered adapter family names."""
    return [a.family for a in _ADAPTER_REGISTRY]
