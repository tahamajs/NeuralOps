"""Inference subpackage for NeuralOps."""
from __future__ import annotations

from neuralops.inference.paged_cache import PagedKVCache
from neuralops.inference.engine import (
    NeuralOpsInferenceEngine,
    stream_generate,
    speculative_generate,
    beam_search_generate,
    ContinuousBatcher,
)

__all__ = [
    "PagedKVCache",
    "NeuralOpsInferenceEngine",
    "stream_generate",
    "speculative_generate",
    "beam_search_generate",
    "ContinuousBatcher",
]

# Auto-wired package members
from neuralops.inference import engine
from neuralops.inference import paged_cache
