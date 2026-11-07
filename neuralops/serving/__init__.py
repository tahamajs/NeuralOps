#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Serving Subpackage.

Canonical interface for NeuralOps serving: OpenAI-compatible FastAPI serving engine,
paged KV-cache management, and specialized export adapters.
"""

from __future__ import annotations

from neuralops.export.serving import TGIAdapter, VLLMAdapter, export_serving_bundle
from neuralops.integrations.serving import create_openai_serving_app
from neuralops.serving.paged_cache_manager import PagedKVCache, PagedKVCacheManager, PagedStateTable
from neuralops.serving.request_batcher import (
    TGIRequest,
    TGIRequestBatch,
    TGIServingAdapter,
    TGIStreamingIterator,
)

__all__ = [
    "create_openai_serving_app",
    "PagedKVCache",
    "PagedKVCacheManager",
    "PagedStateTable",
    "VLLMAdapter",
    "TGIAdapter",
    "export_serving_bundle",
    "TGIRequest",
    "TGIRequestBatch",
    "TGIServingAdapter",
    "TGIStreamingIterator",
]
