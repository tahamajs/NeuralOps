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
    gqa,
    linear_attn,
    low_rank,
    sparse_attn,
    state_update,
)
from neuralops.core.triton_kernels.op_registry import (
    compilable_linear_attn,
    compilable_sparse_attn,
    compilable_low_rank,
    register_custom_op_meta_shape,
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
    from neuralops.core.triton_kernels import fms_scan
    from neuralops.core.triton_kernels.fms_scan import fms_scan_triton
except Exception:
    fms_scan = None
    fms_scan_triton = None

try:
    from neuralops.core.triton_kernels import gdn_scan
    from neuralops.core.triton_kernels.gdn_scan import gdn_scan_triton
except Exception:
    gdn_scan = None
    gdn_scan_triton = None

try:
    from neuralops.core.triton_kernels import gla_scan
    from neuralops.core.triton_kernels.gla_scan import gla_scan_triton
except Exception:
    gla_scan = None
    gla_scan_triton = None

try:
    from neuralops.core.triton_kernels import hgrn2_scan
    from neuralops.core.triton_kernels.hgrn2_scan import hgrn2_scan_triton
except Exception:
    hgrn2_scan = None
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
    "Autotuner",
    "autotune_kernel",
    "check_triton_suitability",
    "log_fallback_reason",
]

# Auto-wired package members
from neuralops.core.triton_kernels import _common
from neuralops.core.triton_kernels import _types
from neuralops.core.triton_kernels import autotuner
from neuralops.core.triton_kernels import block_sparse
from neuralops.core.triton_kernels import cosformer_triton_kernel
from neuralops.core.triton_kernels import elementwise
from neuralops.core.triton_kernels import flash_decoding
from neuralops.core.triton_kernels import fused_rope
from neuralops.core.triton_kernels import gqa
from neuralops.core.triton_kernels import linear_attn
from neuralops.core.triton_kernels import low_rank
from neuralops.core.triton_kernels import paged_attention
from neuralops.core.triton_kernels import selective_scan
from neuralops.core.triton_kernels import sliding_window
from neuralops.core.triton_kernels import softplus
from neuralops.core.triton_kernels import sparse_attn
from neuralops.core.triton_kernels import state_update
from neuralops.core.triton_kernels import varlen_attention
