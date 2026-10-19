# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
NeuralOps Distributed Subpackage.
"""

from __future__ import annotations

from neuralops.distributed.tensor_parallel import (
    SimulatedTPGroup,
    ColumnParallelLinear,
    RowParallelLinear,
    RankParallelSVDLinear,
    RowParallelSVDLinear,
    ColumnParallelSVDLinear,
    TensorParallelRecurrentAttention,
    shard_model_tensor_parallel,
    copy_to_model_parallel_region,
    reduce_from_model_parallel_region,
)
from neuralops.distributed.context_parallel import (
    compute_chunk_prefix_scan,
    ContextParallelRecurrentScan,
    ContextParallelLinearAttention,
    ring_attention_chunk_exchange,
    RingLinearAttention,
)

__all__ = [
    "SimulatedTPGroup",
    "ColumnParallelLinear",
    "RowParallelLinear",
    "RankParallelSVDLinear",
    "RowParallelSVDLinear",
    "ColumnParallelSVDLinear",
    "TensorParallelRecurrentAttention",
    "shard_model_tensor_parallel",
    "copy_to_model_parallel_region",
    "reduce_from_model_parallel_region",
    "compute_chunk_prefix_scan",
    "ContextParallelRecurrentScan",
    "ContextParallelLinearAttention",
    "ring_attention_chunk_exchange",
    "RingLinearAttention",
]
