#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Text Generation Inference (TGI) Integration Stub

TGI runtime integration is deferred to NeuralOps v2.0.
For the request batcher and continuous generation adapter, see:
    neuralops.serving.request_batcher
"""

from __future__ import annotations

import logging
from typing import Any

from neuralops.core.errors import IntegrationNotImplementedError

logger = logging.getLogger("NeuralOps.TGI")


def register_neuralops_tgi_plugin(*args: Any, **kwargs: Any) -> Any:
    """Register NeuralOps plugin within TGI runtime (deferred to v2.0)."""
    raise IntegrationNotImplementedError(
        "TGI runtime integration is not implemented in NeuralOps v1.6.0 and is deferred to v2.0. "
        "For request batching and generation, use neuralops.serving.request_batcher. "
        "See docs/roadmap/TGI_PLAN.md for details."
    )


class TGIRequest:
    """Stub for TGIRequest. Moved to neuralops.serving.request_batcher."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise IntegrationNotImplementedError(
            "TGIRequest in neuralops.integrations.tgi is deferred to v2.0. "
            "For the working implementation, use neuralops.serving.request_batcher.TGIRequest. "
            "See docs/roadmap/TGI_PLAN.md for details."
        )


class TGIRequestBatch:
    """Stub for TGIRequestBatch. Moved to neuralops.serving.request_batcher."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise IntegrationNotImplementedError(
            "TGIRequestBatch in neuralops.integrations.tgi is deferred to v2.0. "
            "For the working implementation, use neuralops.serving.request_batcher.TGIRequestBatch. "
            "See docs/roadmap/TGI_PLAN.md for details."
        )


class TGIStreamingIterator:
    """Stub for TGIStreamingIterator. Moved to neuralops.serving.request_batcher."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise IntegrationNotImplementedError(
            "TGIStreamingIterator in neuralops.integrations.tgi is deferred to v2.0. "
            "For the working implementation, use neuralops.serving.request_batcher.TGIStreamingIterator. "
            "See docs/roadmap/TGI_PLAN.md for details."
        )


class TGIServingAdapter:
    """Stub for TGIServingAdapter. Moved to neuralops.serving.request_batcher."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise IntegrationNotImplementedError(
            "TGIServingAdapter in neuralops.integrations.tgi is deferred to v2.0. "
            "For the working implementation, use neuralops.serving.request_batcher.TGIServingAdapter. "
            "See docs/roadmap/TGI_PLAN.md for details."
        )


TGIRequestSlot = TGIRequest

__all__ = [
    "register_neuralops_tgi_plugin",
    "TGIRequest",
    "TGIRequestSlot",
    "TGIRequestBatch",
    "TGIStreamingIterator",
    "TGIServingAdapter",
]
