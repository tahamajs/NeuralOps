#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Serving Paged Cache Manager

Canonical entrypoint and compatibility alias for PagedKVCache, PagedKVCacheManager,
and dynamic memory management routines in serving deployments.
"""

from __future__ import annotations

from neuralops.inference.paged_cache import (
    PagedKVCache,
    PagedKVCacheManager,
    PagedStateTable,
)

__all__ = ["PagedKVCache", "PagedKVCacheManager", "PagedStateTable"]
