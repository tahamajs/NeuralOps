# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
NeuralOps Inference Subpackage.
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

from neuralops.inference import engine
from neuralops.inference import paged_cache
from neuralops.inference import speculative_decoding

