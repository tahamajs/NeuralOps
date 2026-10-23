#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
SGLang Integration Stub

SGLang runtime integration is deferred to NeuralOps v2.0.
For the standalone Radix Trie prefix cache implementation, see:
    neuralops.integrations.radix_prefix_cache
"""

from __future__ import annotations

import logging
from typing import Any

from neuralops.core.errors import IntegrationNotImplementedError

logger = logging.getLogger("NeuralOps.SGLang")


def register_neuralops_sglang_plugin(*args: Any, **kwargs: Any) -> Any:
    """Register NeuralOps plugin within SGLang runtime (deferred to v2.0)."""
    raise IntegrationNotImplementedError(
        "SGLang runtime integration is not implemented in NeuralOps v1.6.0 and is deferred to v2.0. "
        "For standalone prefix caching, use neuralops.integrations.radix_prefix_cache. "
        "See docs/roadmap/SGLANG_PLAN.md for details."
    )


class RadixPrefixNode:
    """Stub for RadixPrefixNode. Moved to neuralops.integrations.radix_prefix_cache."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise IntegrationNotImplementedError(
            "RadixPrefixNode in neuralops.integrations.sglang is deferred to v2.0. "
            "For the working implementation, use neuralops.integrations.radix_prefix_cache.RadixPrefixNode. "
            "See docs/roadmap/SGLANG_PLAN.md for details."
        )


class SGLangRadixAdapter:
    """Stub for SGLangRadixAdapter. Moved to neuralops.integrations.radix_prefix_cache."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise IntegrationNotImplementedError(
            "SGLangRadixAdapter in neuralops.integrations.sglang is deferred to v2.0. "
            "For the working implementation, use neuralops.integrations.radix_prefix_cache.SGLangRadixAdapter. "
            "See docs/roadmap/SGLANG_PLAN.md for details."
        )


__all__ = [
    "register_neuralops_sglang_plugin",
    "RadixPrefixNode",
    "SGLangRadixAdapter",
]
