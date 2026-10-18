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
)
# Catalog of all Triton Kernels across NeuralOps
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
    # Flagship Hardware Kernels (v1.4.0 Extended Suite)
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
    # 27 Categorized Kernel Modules
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
    # 13 Standalone Flagship Modules
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
    # Root Bridge Modules
    "fast_linear_attention",
    "LinearAttentionTriton",
    "triton_block_sparse_attention",
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
    fused_dsa_sparse_attn_triton,
    fused_dsa_sparse_attn_pytorch_ref,
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
from neuralops.core.triton_kernels.differential_attention import (
    triton_differential_attention,
    triton_differential_attention_pytorch_ref,
    TritonDifferentialAttentionFunction,
)
from neuralops.core.triton_kernels.hydra_attention import (
    triton_hydra_attention,
    triton_hydra_attention_pytorch_ref,
    TritonHydraAttentionFunction,
)
from neuralops.core.triton_kernels.retnet_retention import (
    triton_multi_scale_retention,
    triton_retnet_retention,
    triton_multi_scale_retention_pytorch_ref,
    triton_retnet_retention_pytorch_ref,
    TritonRetNetRetentionFunction,
)
from neuralops.core.triton_kernels.dynamic_sparse_attention import (
    triton_dynamic_sparse_attention,
    triton_dynamic_sparse_attention_pytorch_ref,
    TritonDynamicSparseAttentionFunction,
)
from neuralops.core.triton_kernels.cur_linear import (
    triton_cur_linear,
    triton_cur_linear_pytorch_ref,
    TritonCURLinearFunction,
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
from neuralops.core.triton_kernels import differential_attention
from neuralops.core.triton_kernels import hydra_attention
from neuralops.core.triton_kernels import retnet_retention
from neuralops.core.triton_kernels import dynamic_sparse_attention
from neuralops.core.triton_kernels import cur_linear
from neuralops.core.triton_kernels import varlen_attention
from neuralops.core.triton_kernels import hardware_profile
from neuralops.core.triton_kernels import op_registry
from neuralops.core.triton_kernels import fused_ce as fused_ce_module
from neuralops.core.triton_kernels import fused_kd as fused_kd_module
from neuralops.core.triton_kernels import fused_norm_linear as fused_norm_linear_module

# Root Bridge Modules
from neuralops.core.triton_linear_attn import fast_linear_attention, LinearAttentionTriton
from neuralops.core.triton_sparse_attn import triton_block_sparse_attention


def __getattr__(name: str) -> Any:
    """Dynamically resolve flagship standalone modules from neuralops.kernels.triton."""
    try:
        import neuralops.kernels.triton as nkt

        if hasattr(nkt, name):
            val = getattr(nkt, name)
            globals()[name] = val
            return val
    except Exception:
        pass
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> list[str]:
    """Execute __dir__ operation."""
    return sorted(list(set(globals().keys()) | set(__all__)))
