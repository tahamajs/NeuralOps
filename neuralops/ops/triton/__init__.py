#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
NeuralOps Core Triton Kernels Hardware Operators.
"""

from __future__ import annotations

from typing import Any

from neuralops.ops.triton import (
    gqa,
    linear_attn,
    low_rank,
    sparse_attn,
    state_update,
)
from neuralops.ops.triton.op_registry import (
    compilable_linear_attn,
    compilable_sparse_attn,
    compilable_low_rank,
    register_custom_op_meta_shape,
)
from neuralops.ops.triton._common import (
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
    from neuralops.ops.triton.softplus import softplus
except Exception:
    import torch.nn.functional as F

    softplus = F.softplus

try:
    from neuralops.ops.triton.selective_scan import selective_scan_fn
except Exception:
    selective_scan_fn = None

try:
    from neuralops.ops.triton import fms_scan
    from neuralops.ops.triton.fms_scan import fms_scan_triton
except Exception:
    fms_scan = None
    fms_scan_triton = None

try:
    from neuralops.ops.triton import gdn_scan
    from neuralops.ops.triton.gdn_scan import gdn_scan_triton
except Exception:
    gdn_scan = None
    gdn_scan_triton = None

try:
    from neuralops.ops.triton import gla_scan
    from neuralops.ops.triton.gla_scan import gla_scan_triton
except Exception:
    gla_scan = None
    gla_scan_triton = None

try:
    from neuralops.ops.triton import hgrn2_scan
    from neuralops.ops.triton.hgrn2_scan import hgrn2_scan_triton
except Exception:
    hgrn2_scan = None
    hgrn2_scan_triton = None

from neuralops.ops.triton.fused_ce import (
    FusedCrossEntropyLoss,
    fused_cross_entropy,
    fused_cross_entropy_pytorch,
    FusedLinearCrossEntropy,
    FusedLinearCrossEntropyFunction,
    fused_linear_cross_entropy,
)
from neuralops.ops.triton.fused_kd import (
    FusedKDLoss,
    fused_kd_loss,
    fused_kd_loss_pytorch,
    FusedLinearKLDivergence,
    FusedLinearKLDivergenceFunction,
    fused_linear_kl_divergence,
)
from neuralops.ops.triton.fused_norm_linear import (
    FusedRMSNormLinear,
    fused_norm_linear,
    fused_norm_linear_pytorch,
    fused_rmsnorm_linear,
    fused_rmsnorm_gate,
)
fused_rmsnorm_gate_triton = fused_rmsnorm_gate
TRITON_STANDALONE_KERNELS: list[str] = [
    "block_sparse_streaming_attn",
    "chunked_delta_gate_scan",
    "cosformer_linear_attention",
    "fused_act_scale_quant_gemm",
    "fused_hadamard_w4a16_gemm",
    "fused_kronecker_matmul",
    "fused_latent_kv_cache_attention",
    "fused_lowrank_gemm",
    "hgrn2_fused_bwd_scan",
    "lut_codebook_dequant_gemm",
    "rpca_sparse_gemm",
    "spectral_entropy_svd_gate",
    "strided_tt",
]

TRITON_CORE_MODULES: list[str] = [
    "block_sparse",
    "cosformer_triton_kernel",
    "cur_linear",
    "differential_attention",
    "dynamic_sparse_attention",
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
    "hydra_attention",
    "linear_attn",
    "low_rank",
    "paged_attention",
    "retnet_retention",
    "selective_scan",
    "sliding_window",
    "softplus",
    "sparse_attn",
    "state_update",
    "varlen_attention",
]

TRITON_ROOT_MODULES: list[str] = [
    "triton_linear_attn",
    "triton_sparse_attn",
]

TRITON_EXTENDED_KERNELS: list[str] = [
    "fused_swiglu",
    "fused_rmsnorm",
    "fused_rotary_emb",
    "fused_alibi_attention",
    "fused_sink_token_attention",
    "fused_decay_scan",
    "fused_hadamard_transform",
    "fused_int4_dequant",
    "fused_fp8_e4m3_gemm",
    "fused_spectral_dct",
    "fused_chunked_cumsum",
    "fused_cka_alignment",
    "fused_mohawk_homotopy",
    "fused_nullspace_project",
    "fused_householder_reflect",
    "fused_simplex_project",
    "fused_topk_gating",
    "fused_label_smoothing_ce",
    "fused_kl_divergence",
    "fused_lowrank_residual_add",
]

TRITON_KERNEL_CATALOG: list[str] = sorted(list(set(
    TRITON_STANDALONE_KERNELS + TRITON_CORE_MODULES + TRITON_ROOT_MODULES + TRITON_EXTENDED_KERNELS
)))


def list_triton_kernels(category: str | None = None) -> list[str]:
    """Return list of all available Triton kernels in NeuralOps without artificial restrictions.

    Args:
        category: Filter category:
            - None or 'all': All 62 Triton kernels (standalone, core modules, root, and extended).
            - 'standalone': 13 standalone flagship Triton kernels.
            - 'core': 27 core categorized Triton kernel modules.
            - 'root': 2 root bridge Triton modules.
            - 'extended': 20 extended fused Triton kernels.

    Returns:
        Sorted list of kernel names.
    """
    if category == "standalone":
        return list(TRITON_STANDALONE_KERNELS)
    if category == "core":
        return list(TRITON_CORE_MODULES)
    if category == "root":
        return list(TRITON_ROOT_MODULES)
    if category == "extended":
        return list(TRITON_EXTENDED_KERNELS)
    return list(TRITON_KERNEL_CATALOG)


__all__ = [
    "TRITON_STANDALONE_KERNELS",
    "TRITON_CORE_MODULES",
    "TRITON_ROOT_MODULES",
    "TRITON_EXTENDED_KERNELS",
    "TRITON_KERNEL_CATALOG",
    "list_triton_kernels",
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
    "fused_lowrank_gemm",
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
    "fused_dsa_sparse_attn_triton",
    "fused_dsa_sparse_attn_pytorch_ref",
    "triton_hgrn2_fused_bwd_scan",
    "triton_hgrn2_fused_bwd_scan_pytorch_ref",
    "triton_spectral_entropy_svd_gate",
    "triton_spectral_entropy_svd_gate_pytorch_ref",
    "triton_fused_act_scale_quant_gemm",
    "triton_fused_act_scale_quant_gemm_pytorch_ref",
    "triton_fused_latent_kv_cache_attention",
    "triton_fused_latent_kv_cache_attention_pytorch_ref",
    "triton_differential_attention",
    "triton_differential_attention_pytorch_ref",
    "TritonDifferentialAttentionFunction",
    "triton_hydra_attention",
    "triton_hydra_attention_pytorch_ref",
    "TritonHydraAttentionFunction",
    "triton_multi_scale_retention",
    "triton_retnet_retention",
    "triton_multi_scale_retention_pytorch_ref",
    "triton_retnet_retention_pytorch_ref",
    "TritonRetNetRetentionFunction",
    "triton_dynamic_sparse_attention",
    "triton_dynamic_sparse_attention_pytorch_ref",
    "TritonDynamicSparseAttentionFunction",
    "triton_cur_linear",
    "triton_cur_linear_pytorch_ref",
    "TritonCURLinearFunction",
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
    "block_sparse",
    "cosformer_triton_kernel",
    "cur_linear",
    "differential_attention",
    "dynamic_sparse_attention",
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
    "hydra_attention",
    "linear_attn",
    "low_rank",
    "paged_attention",
    "retnet_retention",
    "selective_scan",
    "sliding_window",
    "softplus",
    "sparse_attn",
    "state_update",
    "varlen_attention",
    "autotuner",
    "hardware_profile",
    "op_registry",
    "_common",
    "_types",
    "block_sparse_streaming_attn",
    "chunked_delta_gate_scan",
    "cosformer_linear_attention",
    "fused_act_scale_quant_gemm",
    "fused_hadamard_w4a16_gemm",
    "fused_kronecker_matmul",
    "fused_latent_kv_cache_attention",
    "fused_lowrank_gemm",
    "hgrn2_fused_bwd_scan",
    "lut_codebook_dequant_gemm",
    "rpca_sparse_gemm",
    "spectral_entropy_svd_gate",
    "strided_tt",
    "fast_linear_attention",
    "LinearAttentionTriton",
    "triton_block_sparse_attention",
]

from neuralops.ops.triton.low_rank import (
    fused_lowrank_gemm,
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
from neuralops.ops.triton.fused_norm_linear import (
    triton_fused_hadamard_w4a16_gemm,
    triton_fused_hadamard_w4a16_gemm_pytorch_ref,
    triton_lut_codebook_dequant_gemm,
    triton_lut_codebook_dequant_gemm_pytorch_ref,
    triton_fused_act_scale_quant_gemm,
    triton_fused_act_scale_quant_gemm_pytorch_ref,
)
from neuralops.ops.triton.gdn_scan import (
    triton_chunked_delta_gate_scan,
    triton_chunked_delta_gate_scan_pytorch_ref,
)
from neuralops.ops.triton.cosformer_triton_kernel import (
    triton_cosformer_linear_attention,
    triton_cosformer_linear_attention_pytorch_ref,
)
from neuralops.ops.triton.block_sparse import (
    triton_block_sparse_streaming_attn,
    triton_block_sparse_streaming_attn_pytorch_ref,
    fused_dsa_sparse_attn_triton,
    fused_dsa_sparse_attn_pytorch_ref,
)
from neuralops.ops.triton.hgrn2_scan import (
    triton_hgrn2_fused_bwd_scan,
    triton_hgrn2_fused_bwd_scan_pytorch_ref,
)
from neuralops.ops.triton.linear_attn import (
    triton_fused_latent_kv_cache_attention,
    triton_fused_latent_kv_cache_attention_pytorch_ref,
)
from neuralops.ops.triton.flash_decoding import (
    flash_decoding_attention,
    flash_decoding_stage1_kernel,
    flash_decoding_stage2_kernel,
)
from neuralops.ops.triton.paged_attention import (
    paged_attention_kernel,
    paged_recurrent_state_step_triton,
    paged_kv_evict_compact_triton,
)
from neuralops.ops.triton.fused_rope import (
    fused_rope_embedding_kernel,
    fused_unified_rope_offset_triton,
)
from neuralops.ops.triton.elementwise import (
    fast_hadamard_transform,
    fused_rms_variance_match_triton,
    fused_favor_plus_feature_map_triton,
    fused_entropy_token_pruning_mask_triton,
    fused_aegis_multibranch_gate_norm_triton,
    fused_split_head_hybrid_concat_norm_triton,
)
from neuralops.ops.triton.differential_attention import (
    triton_differential_attention,
    triton_differential_attention_pytorch_ref,
    TritonDifferentialAttentionFunction,
)
from neuralops.ops.triton.hydra_attention import (
    triton_hydra_attention,
    triton_hydra_attention_pytorch_ref,
    TritonHydraAttentionFunction,
)
from neuralops.ops.triton.retnet_retention import (
    triton_multi_scale_retention,
    triton_retnet_retention,
    triton_multi_scale_retention_pytorch_ref,
    triton_retnet_retention_pytorch_ref,
    TritonRetNetRetentionFunction,
)
from neuralops.ops.triton.dynamic_sparse_attention import (
    triton_dynamic_sparse_attention,
    triton_dynamic_sparse_attention_pytorch_ref,
    TritonDynamicSparseAttentionFunction,
)
from neuralops.ops.triton.cur_linear import (
    triton_cur_linear,
    triton_cur_linear_pytorch_ref,
    TritonCURLinearFunction,
)


try:
    from neuralops.ops.triton.gdn_scan import gdn_recurrent_step_triton
except Exception:
    gdn_recurrent_step_triton = None

try:
    from neuralops.ops.triton.gla_scan import gla_recurrent_step_triton
except Exception:
    gla_recurrent_step_triton = None

try:
    from neuralops.ops.triton.autotuner import select_optimal_triton_config
except Exception:
    select_optimal_triton_config = None

from neuralops.ops.triton import _common
from neuralops.ops.triton import _types
from neuralops.ops.triton import autotuner
from neuralops.ops.triton import block_sparse
from neuralops.ops.triton import cosformer_triton_kernel
from neuralops.ops.triton import elementwise
from neuralops.ops.triton import flash_decoding
from neuralops.ops.triton import fused_rope
from neuralops.ops.triton import paged_attention
from neuralops.ops.triton import selective_scan
from neuralops.ops.triton import sliding_window
from neuralops.ops.triton import softplus
from neuralops.ops.triton import differential_attention
from neuralops.ops.triton import hydra_attention
from neuralops.ops.triton import retnet_retention
from neuralops.ops.triton import dynamic_sparse_attention
from neuralops.ops.triton import cur_linear
from neuralops.ops.triton import varlen_attention
from neuralops.ops.triton import hardware_profile
from neuralops.ops.triton import op_registry

from neuralops.core.triton_linear_attn import fast_linear_attention, LinearAttentionTriton
from neuralops.core.triton_sparse_attn import triton_block_sparse_attention


def __getattr__(name: str) -> Any:
    """Resolve specialized Triton operators from the canonical standalone package."""
    try:
        from neuralops.ops.triton import standalone

        val = getattr(standalone, name)
        globals()[name] = val
        return val
    except Exception:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'") from None


def __dir__() -> list[str]:
    """Execute __dir__ operation."""
    return sorted(list(set(globals().keys()) | set(__all__)))
