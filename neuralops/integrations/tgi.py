#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Text Generation Inference (TGI) Serving Integration Bridge (v2.0 Roadmap)

Architectural Role & Forward Contract (NeuralOps v1.6.0 -> v2.0):
This module defines the integration bridge for TGI runtime serving.

In NeuralOps v1.6.0:
1. TGI Runtime Plugin: Deep Rust/C++ runtime integration with TGI is scheduled for v2.0.
   `register_neuralops_tgi_plugin` fails closed with `IntegrationNotImplementedError`
   specifying the roadmap bridge contract (`docs/roadmap/TGI_PLAN.md`).
2. Request Batcher & Continuous Generation: Lightweight continuous batching, dynamic sequence
   slotting, and streaming iterators are re-exported here from `neuralops.serving.request_batcher`.
"""

from __future__ import annotations

import logging
from typing import Any

from neuralops.core.errors import IntegrationNotImplementedError
from neuralops.serving.request_batcher import (
    TGIRequest,
    TGIRequestSlot,
    TGIRequestBatch,
    TGIStreamingIterator,
    TGIServingAdapter,
)

logger = logging.getLogger("NeuralOps.TGI")


def register_neuralops_tgi_plugin(*args: Any, **kwargs: Any) -> Any:
    """Register NeuralOps plugin within TGI runtime (deferred to v2.0)."""
    raise IntegrationNotImplementedError(
        "TGI runtime engine plugin registration is not implemented in NeuralOps v1.6.0 "
        "and is deferred to v2.0. For request batching, dynamic sequence slotting, "
        "and continuous generation, use `neuralops.serving.request_batcher`. "
        "See docs/roadmap/TGI_PLAN.md for details."
    )


__all__ = [
    "register_neuralops_tgi_plugin",
    "TGIRequest",
    "TGIRequestSlot",
    "TGIRequestBatch",
    "TGIStreamingIterator",
    "TGIServingAdapter",
]
