#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Registry Bridge for FLA Integration.

Maintains EXTENSION_ATTENTION_REGISTRY completely isolated from canonical
ATTENTION_REGISTRY to protect repository invariants (AGENTS.md §8).
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

SUPPORTED_FLA_ATTENTIONS = (
    "fla:gla",
    "fla:retnet",
    "fla:gated_deltanet",
    "fla:rwkv6",
    "fla:hgrn2",
)

# Isolated extension dictionary preserving ATTENTION_REGISTRY invariant (exactly 131)
EXTENSION_ATTENTION_REGISTRY: dict[str, Any] = {}


def register_fla_attention(name: str, fla_cls: Optional[Any] = None) -> None:
    """
    Register an FLA-backed attention module into EXTENSION_ATTENTION_REGISTRY.

    Args:
        name: Attention identifier (must start with 'fla:').
        fla_cls: Optional FLA attention class or constructor.
    """
    if not name.startswith("fla:"):
        raise ValueError(f"FLA extension mechanisms must begin with 'fla:' prefix, got '{name}'")
    EXTENSION_ATTENTION_REGISTRY[name] = fla_cls
    logger.debug("Registered FLA extension attention: %s", name)


def list_fla_attentions() -> List[str]:
    """
    Return sorted list of available FLA-backed attention mechanisms.
    Returns empty list if FLA is not installed in the current environment.
    """
    from neuralops.integrations.fla import fla_is_available

    if not fla_is_available():
        return []
    return sorted(list(EXTENSION_ATTENTION_REGISTRY.keys()))


# Pre-register standard supported FLA aliases into the extension registry
for _alias in SUPPORTED_FLA_ATTENTIONS:
    register_fla_attention(_alias, None)
