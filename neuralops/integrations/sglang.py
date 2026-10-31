#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
SGLang Serving Integration Bridge (v2.0 Roadmap)

Architectural Role & Forward Contract (NeuralOps v1.6.0 -> v2.0):
This module defines the integration bridge for SGLang runtime serving.

In NeuralOps v1.6.0:
1. SGLang Runtime Engine Plugin: Deep integration with SGLang's internal scheduler,
   KV cache page allocator, and C++/Triton dispatch hooks is scheduled for v2.0.
   `register_neuralops_sglang_plugin` fails closed with `IntegrationNotImplementedError`
   specifying the roadmap bridge contract (`docs/roadmap/SGLANG_PLAN.md`).
2. Standalone Prefix Caching: Fully functional Radix Trie prefix caching is provided
   via `RadixPrefixNode` and `SGLangRadixAdapter` (re-exported here from
   `neuralops.integrations.radix_prefix_cache`), supporting exact/longest prefix matching,
   dynamic node branching, token reuse, and LRU eviction.
"""

from __future__ import annotations

import logging
from typing import Any

from neuralops.core.errors import IntegrationNotImplementedError
from neuralops.integrations.radix_prefix_cache import (
    RadixPrefixNode,
    SGLangRadixAdapter,
)

logger = logging.getLogger("NeuralOps.SGLang")


def register_neuralops_sglang_plugin(*args: Any, **kwargs: Any) -> Any:
    """Register NeuralOps plugin within SGLang runtime (deferred to v2.0)."""
    raise IntegrationNotImplementedError(
        "SGLang runtime engine plugin registration is not implemented in NeuralOps v1.6.0 "
        "and is deferred to v2.0. For standalone prefix caching and Radix Trie management, "
        "use `neuralops.integrations.radix_prefix_cache` or `SGLangRadixAdapter`. "
        "See docs/roadmap/SGLANG_PLAN.md for details."
    )


__all__ = [
    "register_neuralops_sglang_plugin",
    "RadixPrefixNode",
    "SGLangRadixAdapter",
]
