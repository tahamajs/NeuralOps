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

from neuralops.distributed.context import DistributedContext
from neuralops.distributed.context_parallel import (
    ContextParallelLinearAttention,
    ContextParallelRecurrentScan,
    RingLinearAttention,
    compute_chunk_prefix_scan,
    ring_attention_chunk_exchange,
)
from neuralops.distributed.errors import (
    E_DIST_COMM_FAILED,
    E_DIST_INIT_FAILED,
    E_DIST_NOT_INITIALIZED,
    E_FSDP2_CONFIG_FAILED,
    E_FSDP2_NOT_INITIALIZED,
    DistributedError,
)
from neuralops.distributed.fsdp2_distill import FSDP2Distillation
from neuralops.distributed.tensor_parallel import (
    ColumnParallelLinear,
    ColumnParallelSVDLinear,
    RankParallelSVDLinear,
    RowParallelLinear,
    RowParallelSVDLinear,
    SimulatedTPGroup,
    TensorParallelRecurrentAttention,
    copy_to_model_parallel_region,
    reduce_from_model_parallel_region,
    shard_model_tensor_parallel,
)

__all__ = [
    "DistributedContext",
    "DistributedError",
    "E_DIST_INIT_FAILED",
    "E_DIST_NOT_INITIALIZED",
    "E_DIST_COMM_FAILED",
    "E_FSDP2_CONFIG_FAILED",
    "E_FSDP2_NOT_INITIALIZED",
    "FSDP2Distillation",
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
