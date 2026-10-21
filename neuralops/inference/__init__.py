#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Lazy public exports for autoregressive inference utilities."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "PagedKVCache": ("paged_cache", "PagedKVCache"),
    "PagedStateTable": ("paged_cache", "PagedStateTable"),
    "NeuralOpsInferenceEngine": ("engine", "NeuralOpsInferenceEngine"),
    "stream_generate": ("engine", "stream_generate"),
    "speculative_generate": ("engine", "speculative_generate"),
    "beam_search_generate": ("engine", "beam_search_generate"),
    "ContinuousBatcher": ("engine", "ContinuousBatcher"),
    "SelfSpeculativeDecoder": ("speculative_decoding", "SelfSpeculativeDecoder"),
    "SpeculativeDecodeStats": ("speculative_decoding", "SpeculativeDecodeStats"),
    "gamma_sweep_harness": ("speculative_decoding", "gamma_sweep_harness"),
    "CUDAGraphDecodeRunner": ("neuralops.models.cuda_graph", "CUDAGraphDecodeRunner"),
}
_MODULES = {"engine", "paged_cache", "speculative_decoding"}

__all__ = [*_EXPORTS, *_MODULES]


def __getattr__(name: str):
    """Load decoding and CUDA-graph components only when requested."""
    if name in _EXPORTS:
        module_name, attribute = _EXPORTS[name]
        target = module_name if module_name.startswith("neuralops.") else f"{__name__}.{module_name}"
        value = getattr(import_module(target), attribute)
    elif name in _MODULES:
        value = import_module(f"{__name__}.{name}")
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
