#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Serving Cache Compatibility Module.
Re-exports PagedKVCache and related abstractions from neuralops.inference.paged_cache.
"""

from __future__ import annotations

from neuralops.inference.paged_cache import PagedKVCache, PagedKVCacheBlock, PagedKVCacheManager

__all__ = ["PagedKVCache", "PagedKVCacheBlock", "PagedKVCacheManager"]
