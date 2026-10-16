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
    FORCE_PYTORCH_FALLBACK,
    gqa_pytorch_fallback,
    linear_attn_pytorch_fallback,
    low_rank_pytorch_fallback,
    sparse_attn_pytorch_fallback,
    state_update_pytorch_fallback,
    triton_dispatch,
    Autotuner,
    autotune_kernel,
    check_triton_suitability,
    log_fallback_reason,
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

from neuralops.core.triton_kernels.fused_ce import (
    FusedCrossEntropyLoss,
    fused_cross_entropy,
    fused_cross_entropy_pytorch,
    FusedLinearCrossEntropy,
    FusedLinearCrossEntropyFunction,
    fused_linear_cross_entropy,
)
from neuralops.core.triton_kernels.fused_kd import (
    FusedKDLoss,
    fused_kd_loss,
    fused_kd_loss_pytorch,
    FusedLinearKLDivergence,
    FusedLinearKLDivergenceFunction,
    fused_linear_kl_divergence,
)
from neuralops.core.triton_kernels.fused_norm_linear import (
    FusedRMSNormLinear,
    fused_norm_linear,
    fused_norm_linear_pytorch,
    fused_rmsnorm_linear,
    fused_rmsnorm_gate,
    fused_rmsnorm_gate_triton,
)

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
    "FusedCrossEntropyLoss",
    "fused_cross_entropy",
    "fused_cross_entropy_pytorch",
    "FusedLinearCrossEntropy",
    "FusedLinearCrossEntropyFunction",
    "fused_linear_cross_entropy",
    "FusedKDLoss",
    "fused_kd_loss",
    "fused_kd_loss_pytorch",
    "FusedLinearKLDivergence",
    "FusedLinearKLDivergenceFunction",
    "fused_linear_kl_divergence",
    "FusedRMSNormLinear",
    "fused_norm_linear",
    "fused_norm_linear_pytorch",
    "fused_rmsnorm_linear",
    "fused_rmsnorm_gate",
    "fused_rmsnorm_gate_triton",
    "gdn_recurrent_step_triton",
    "gla_recurrent_step_triton",
    "select_optimal_triton_config",
    # 13 Flagship Triton Kernels
    "triton_fused_lowrank_gated_gemm",
    "triton_fused_lowrank_gated_gemm_pytorch_ref",
    "triton_fused_lowrank_sparse_gemm",
    "triton_fused_lowrank_sparse_gemm_pytorch_ref",
    "triton_strided_tt_contraction",
    "triton_strided_tt_linear",
    "triton_strided_tt_contraction_pytorch_ref",
    "triton_fused_hadamard_w4a16_gemm",
    "triton_fused_hadamard_w4a16_gemm_pytorch_ref",
    "triton_lut_codebook_dequant_gemm",
    "triton_lut_codebook_dequant_gemm_pytorch_ref",
    "triton_fused_kronecker_matmul",
    "triton_fused_kronecker_matmul_pytorch_ref",
    "triton_chunked_delta_gate_scan",
    "triton_chunked_delta_gate_scan_pytorch_ref",
    "triton_cosformer_linear_attention",
    "triton_cosformer_linear_attention_pytorch_ref",
    "triton_block_sparse_streaming_attn",
    "triton_block_sparse_streaming_attn_pytorch_ref",
    "triton_hgrn2_fused_bwd_scan",
    "triton_hgrn2_fused_bwd_scan_pytorch_ref",
    "triton_spectral_entropy_svd_gate",
    "triton_spectral_entropy_svd_gate_pytorch_ref",
    "triton_fused_act_scale_quant_gemm",
    "triton_fused_act_scale_quant_gemm_pytorch_ref",
    "triton_fused_latent_kv_cache_attention",
    "triton_fused_latent_kv_cache_attention_pytorch_ref",
    # 10 High-Performance Production Kernels
    "flash_decoding_attention",
    "flash_decoding_stage1_kernel",
    "flash_decoding_stage2_kernel",
    "paged_attention_kernel",
    "paged_recurrent_state_step_triton",
    "paged_kv_evict_compact_triton",
    "fused_rope_embedding_kernel",
    "fused_unified_rope_offset_triton",
    "fast_hadamard_transform",
    "fused_rms_variance_match_triton",
    "fused_favor_plus_feature_map_triton",
    "fused_entropy_token_pruning_mask_triton",
    "fused_aegis_multibranch_gate_norm_triton",
    "fused_split_head_hybrid_concat_norm_triton",
]

# Import 13 Flagship Triton Kernels
from neuralops.core.triton_kernels.low_rank import (
    triton_fused_lowrank_gated_gemm,
    triton_fused_lowrank_gated_gemm_pytorch_ref,
    triton_fused_lowrank_sparse_gemm,
    triton_fused_lowrank_sparse_gemm_pytorch_ref,
    triton_strided_tt_contraction,
    triton_strided_tt_linear,
    triton_strided_tt_contraction_pytorch_ref,
    triton_fused_kronecker_matmul,
    triton_fused_kronecker_matmul_pytorch_ref,
    triton_spectral_entropy_svd_gate,
    triton_spectral_entropy_svd_gate_pytorch_ref,
)
from neuralops.core.triton_kernels.fused_norm_linear import (
    triton_fused_hadamard_w4a16_gemm,
    triton_fused_hadamard_w4a16_gemm_pytorch_ref,
    triton_lut_codebook_dequant_gemm,
    triton_lut_codebook_dequant_gemm_pytorch_ref,
    triton_fused_act_scale_quant_gemm,
    triton_fused_act_scale_quant_gemm_pytorch_ref,
)
from neuralops.core.triton_kernels.gdn_scan import (
    triton_chunked_delta_gate_scan,
    triton_chunked_delta_gate_scan_pytorch_ref,
)
from neuralops.core.triton_kernels.cosformer_triton_kernel import (
    triton_cosformer_linear_attention,
    triton_cosformer_linear_attention_pytorch_ref,
)
from neuralops.core.triton_kernels.block_sparse import (
    triton_block_sparse_streaming_attn,
    triton_block_sparse_streaming_attn_pytorch_ref,
)
from neuralops.core.triton_kernels.hgrn2_scan import (
    triton_hgrn2_fused_bwd_scan,
    triton_hgrn2_fused_bwd_scan_pytorch_ref,
)
from neuralops.core.triton_kernels.linear_attn import (
    triton_fused_latent_kv_cache_attention,
    triton_fused_latent_kv_cache_attention_pytorch_ref,
)
from neuralops.core.triton_kernels.flash_decoding import (
    flash_decoding_attention,
    flash_decoding_stage1_kernel,
    flash_decoding_stage2_kernel,
)
from neuralops.core.triton_kernels.paged_attention import (
    paged_attention_kernel,
    paged_recurrent_state_step_triton,
    paged_kv_evict_compact_triton,
)
from neuralops.core.triton_kernels.fused_rope import (
    fused_rope_embedding_kernel,
    fused_unified_rope_offset_triton,
)
from neuralops.core.triton_kernels.elementwise import (
    fast_hadamard_transform,
    fused_rms_variance_match_triton,
    fused_favor_plus_feature_map_triton,
    fused_entropy_token_pruning_mask_triton,
    fused_aegis_multibranch_gate_norm_triton,
    fused_split_head_hybrid_concat_norm_triton,
)


try:
    from neuralops.core.triton_kernels.gdn_scan import gdn_recurrent_step_triton
except Exception:
    gdn_recurrent_step_triton = None

try:
    from neuralops.core.triton_kernels.gla_scan import gla_recurrent_step_triton
except Exception:
    gla_recurrent_step_triton = None

try:
    from neuralops.core.triton_kernels.autotuner import select_optimal_triton_config
except Exception:
    select_optimal_triton_config = None

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
