#!/usr/bin/env python3
"""
Shared Triton Kernels for Attention and Linear Architectures.

Provides high-performance fused GPU kernels with transparent PyTorch CPU/CUDA fallbacks:
  - linear_attn: Fused chunked causal linear attention and bidirectional reduction
  - sparse_attn: Block-sparse attention
  - low_rank: Bottleneck low-rank projections
  - state_update: SSM / RWKV recurrence
  - gqa: Grouped Query Attention head expansion
  - selective_scan: Triton-accelerated selective scan for Mamba-1
"""

from __future__ import annotations

from framework.core.triton_kernels import (
    gqa,
    linear_attn,
    low_rank,
    sparse_attn,
    state_update,
)
from framework.core.triton_kernels._common import (
    HAS_TRITON,
    gqa_pytorch_fallback,
    linear_attn_pytorch_fallback,
    low_rank_pytorch_fallback,
    sparse_attn_pytorch_fallback,
    state_update_pytorch_fallback,
    triton_dispatch,
)

try:
    from framework.core.triton_kernels.softplus import softplus
except Exception:
    import torch.nn.functional as F

    softplus = F.softplus

try:
    from framework.core.triton_kernels.selective_scan import selective_scan_fn
except Exception:
    selective_scan_fn = None

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
    "softplus",
    "selective_scan_fn",
]
