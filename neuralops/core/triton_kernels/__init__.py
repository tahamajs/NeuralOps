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
  - fms_scan: Factorised Matrix State scan
  - gdn_scan: Gated DeltaNet recurrent scan
  - gla_scan: Gated Linear Attention recurrent scan
  - hgrn2_scan: HGRN2 recurrent scan
"""

from __future__ import annotations

from neuralops.core.triton_kernels import (
    fms_scan,
    gdn_scan,
    gla_scan,
    gqa,
    hgrn2_scan,
    linear_attn,
    low_rank,
    sparse_attn,
    state_update,
)
from neuralops.core.triton_kernels._common import (
    HAS_TRITON,
    gqa_pytorch_fallback,
    linear_attn_pytorch_fallback,
    low_rank_pytorch_fallback,
    sparse_attn_pytorch_fallback,
    state_update_pytorch_fallback,
    triton_dispatch,
)

try:
    from neuralops.core.triton_kernels.softplus import softplus
except Exception:
    import torch.nn.functional as F

    softplus = F.softplus

try:
    from neuralops.core.triton_kernels.selective_scan import selective_scan_fn
except Exception:
    selective_scan_fn = None

try:
    from neuralops.core.triton_kernels.fms_scan import fms_scan_triton
except Exception:
    fms_scan_triton = None

try:
    from neuralops.core.triton_kernels.gdn_scan import gdn_scan_triton
except Exception:
    gdn_scan_triton = None

try:
    from neuralops.core.triton_kernels.gla_scan import gla_scan_triton
except Exception:
    gla_scan_triton = None

try:
    from neuralops.core.triton_kernels.hgrn2_scan import hgrn2_scan_triton
except Exception:
    hgrn2_scan_triton = None

__all__ = [
    "HAS_TRITON",
    "triton_dispatch",
    "linear_attn",
    "sparse_attn",
    "low_rank",
    "state_update",
    "gqa",
    "fms_scan",
    "gdn_scan",
    "gla_scan",
    "hgrn2_scan",
    "linear_attn_pytorch_fallback",
    "sparse_attn_pytorch_fallback",
    "low_rank_pytorch_fallback",
    "state_update_pytorch_fallback",
    "gqa_pytorch_fallback",
    "softplus",
    "selective_scan_fn",
    "fms_scan_triton",
    "gdn_scan_triton",
    "gla_scan_triton",
    "hgrn2_scan_triton",
]
