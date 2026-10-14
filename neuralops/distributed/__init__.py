"""
NeuralOps Scaled Distributed Engine Package.

Provides:
- Megatron-Style Tensor Parallelism (TP) for Recurrent Attentions and Low-Rank SVD layers.
- Exact 3-Stage Ring / Prefix Context Parallelism (CP) for 1M+ token sequences.
- Distributed compilation and FSDP integration helpers.
"""
from __future__ import annotations

from neuralops.distributed.tensor_parallel import (
    SimulatedTPGroup,
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
)

__all__ = [
    "SimulatedTPGroup",
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
]
