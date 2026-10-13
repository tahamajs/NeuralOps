"""
Canonical `neuralops.kernels` subpackage.
Re-exports ONLY actual computational Triton kernels, triton_dispatch, and HAS_TRITON.
"""

from __future__ import annotations

from neuralops.core.triton_kernels import (
    HAS_TRITON,
    triton_dispatch,
    linear_attn,
    sparse_attn,
    low_rank,
    state_update,
    gqa,
    fms_scan,
    gdn_scan,
    gla_scan,
    hgrn2_scan,
    selective_scan_fn,
    softplus,
)
from neuralops.core.triton_kernels import (
    block_sparse,
    cosformer_triton_kernel,
    elementwise,
    flash_decoding,
    fused_ce,
    fused_kd,
    fused_norm_linear,
    fused_rope,
    paged_attention,
    selective_scan,
    sliding_window,
    varlen_attention,
)

__all__ = [
    "HAS_TRITON",
    "triton_dispatch",
    "block_sparse",
    "cosformer_triton_kernel",
    "elementwise",
    "flash_decoding",
    "fms_scan",
    "fused_ce",
    "fused_kd",
    "fused_norm_linear",
    "fused_rope",
    "gdn_scan",
    "gla_scan",
    "gqa",
    "hgrn2_scan",
    "linear_attn",
    "low_rank",
    "paged_attention",
    "selective_scan",
    "selective_scan_fn",
    "sliding_window",
    "softplus",
    "sparse_attn",
    "state_update",
    "varlen_attention",
]
