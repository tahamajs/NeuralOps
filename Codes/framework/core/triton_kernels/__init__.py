#!/usr/bin/env python3
"""
Shared Triton Kernels for Attention and Linear Architectures.

Provides high-performance fused GPU kernels with transparent PyTorch CPU/CUDA fallbacks:
  - linear_attn: Fused chunked causal linear attention and bidirectional reduction
  - sparse_attn: Block-sparse attention
  - low_rank: Bottleneck low-rank projections
  - state_update: SSM / RWKV recurrence
  - gqa: Grouped Query Attention head expansion
"""
from __future__ import annotations

from framework.core.triton_kernels._common import (
    HAS_TRITON,
    triton_dispatch,
    linear_attn_pytorch_fallback,
    sparse_attn_pytorch_fallback,
    low_rank_pytorch_fallback,
    state_update_pytorch_fallback,
    gqa_pytorch_fallback,
)
from framework.core.triton_kernels import (
    gqa,
    linear_attn,
    low_rank,
    sparse_attn,
    state_update,
)

__all__ = [
    "HAS_TRITON",
    "triton_dispatch",
    "linear_attn",
    "sparse_attn",
    "low_rank",
    "state_update",
    "gqa",
    "linear_attn_pytorch_fallback",
    "sparse_attn_pytorch_fallback",
    "low_rank_pytorch_fallback",
    "state_update_pytorch_fallback",
    "gqa_pytorch_fallback",
]
