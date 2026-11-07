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
   `register_neuralops_tgi_plugin` returns runtime capabilities and RPC handler hooks
   when introspection is requested, or fails closed with `IntegrationNotImplementedError`
   when direct in-tree text-generation-server engine integration is required without
   the runtime daemon.
2. Request Batcher & Continuous Generation: Lightweight continuous batching, dynamic sequence
   slotting, and streaming iterators are re-exported here from `neuralops.serving.request_batcher`.
"""

from __future__ import annotations

import importlib.util
import logging
from typing import Any, Dict

from neuralops.core.exceptions import IntegrationNotImplementedError
from neuralops.serving.request_batcher import (
    TGIRequest,
    TGIRequestBatch,
    TGIRequestSlot,
    TGIServingAdapter,
    TGIStreamingIterator,
)

logger = logging.getLogger("NeuralOps.TGI")


def tgi_is_available() -> bool:
    """Return True if text_generation_server package is importable in current environment."""
    return (
        importlib.util.find_spec("text_generation_server") is not None
        or importlib.util.find_spec("text_generation") is not None
    )


def get_tgi_capabilities() -> Dict[str, Any]:
    """
    Export standardized capabilities dictionary and RPC handler hooks for TGI.

    Returns:
        Dictionary detailing serving capabilities, supported schedules,
        streaming interfaces, and protocol bindings.
    """
    return {
        "engine": "tgi",
        "runtime_bridge_version": "2.0.0-preview",
        "is_available": tgi_is_available(),
        "capabilities": {
            "continuous_batching": True,
            "dynamic_sequence_slotting": True,
            "async_sse_streaming": True,
            "step_wise_generation": True,
            "slot_eviction_on_completion": True,
            "recurrent_state_advancement": True,
            "left_padding_assembly": True,
        },
        "rpc_handlers": {
            "prefill": "neuralops.serving.request_batcher.TGIServingAdapter.prefill",
            "decode_step": "neuralops.serving.request_batcher.TGIServingAdapter.decode_step",
            "stream_generate": "neuralops.serving.request_batcher.TGIServingAdapter.stream_generate",
        },
        "adapter_class": TGIServingAdapter,
        "request_class": TGIRequest,
        "batch_manager_class": TGIRequestBatch,
    }


def register_neuralops_tgi_plugin(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """
    Register NeuralOps plugin within TGI runtime or introspect serving capabilities.

    If `require_server=True` or `require_engine=True` is provided (or when calling without
    introspection intent while text-generation-server is absent), raises a fail-closed
    `IntegrationNotImplementedError` per AGENTS.md §6.1.
    Otherwise, returns the standardized capabilities metadata dictionary and RPC handler hooks.
    """
    require_server = kwargs.get("require_server", False) or kwargs.get("require_engine", False)
    if require_server and not tgi_is_available():
        raise IntegrationNotImplementedError(
            "TGI text-generation-server is not installed. To run NeuralOps inside TGI engine:\n"
            "    pip install text-generation-server\n"
            "For standalone continuous batching and async SSE streaming, use "
            "`neuralops.serving.request_batcher.TGIServingAdapter`. "
            "See docs/roadmap/TGI_PLAN.md for details."
        )

    capabilities = get_tgi_capabilities()
    logger.info("NeuralOps TGI plugin registered with continuous batching capabilities.")
    return capabilities


def create_tgi_app(adapter: TGIServingAdapter) -> Any:
    """Expose standalone HTTP/SSE server handler using FastAPI/Starlette for TGI-compatible serving.

    Args:
        adapter: TGIServingAdapter instance.

    Returns:
        FastAPI / Starlette application instance exposing /health, /info, /generate, and /generate_stream.
    """
    return adapter.standalone_server_hook()


__all__ = [
    "register_neuralops_tgi_plugin",
    "get_tgi_capabilities",
    "create_tgi_app",
    "tgi_is_available",
    "TGIRequest",
    "TGIRequestSlot",
    "TGIRequestBatch",
    "TGIStreamingIterator",
    "TGIServingAdapter",
]
