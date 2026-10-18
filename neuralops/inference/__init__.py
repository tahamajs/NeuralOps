"""
NeuralOps Inference Subpackage.

Author: Taha Majlesi
"""

from __future__ import annotations

from neuralops.inference.paged_cache import PagedKVCache, PagedStateTable
from neuralops.inference.engine import (
    NeuralOpsInferenceEngine,
    stream_generate,
    speculative_generate,
    beam_search_generate,
    ContinuousBatcher,
)

from neuralops.inference.speculative_decoding import (
    SelfSpeculativeDecoder,
    SpeculativeDecodeStats,
    gamma_sweep_harness,
)
from neuralops.models.cuda_graph import CUDAGraphDecodeRunner

__all__ = [
    "PagedKVCache",
    "PagedStateTable",
    "NeuralOpsInferenceEngine",
    "stream_generate",
    "speculative_generate",
    "beam_search_generate",
    "ContinuousBatcher",
    "SelfSpeculativeDecoder",
    "SpeculativeDecodeStats",
    "gamma_sweep_harness",
    "CUDAGraphDecodeRunner",
]

# Auto-wired package members
from neuralops.inference import engine
from neuralops.inference import paged_cache
from neuralops.inference import speculative_decoding

