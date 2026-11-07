#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for PagedKVCacheManager alias consistency (finding F-51 / AGT-16 / AGT-22).
"""

from __future__ import annotations

import pytest
from neuralops.inference.paged_cache import PagedKVCacheManager as InfManager
from neuralops.serving.cache import PagedKVCacheManager as CacheManager
from neuralops.serving.paged_cache_manager import PagedKVCacheManager as ServManager
from neuralops.serving import PagedKVCacheManager as PkgManager


def test_paged_cache_manager_is_the_real_manager():
    """Verify PagedKVCacheManager is the same identical class across all serving and inference paths."""
    assert ServManager is InfManager
    assert ServManager is CacheManager
    assert ServManager is PkgManager
