"""
NeuralOps Kernels Hardware Operators.

Author: Taha Majlesi
"""

from __future__ import annotations

import sys
from typing import Any, List, Optional

# Core Triton & CUDA Subpackages
from neuralops.kernels import triton
from neuralops.kernels import cuda

# Discovery and Kernel Listing APIs
from neuralops.kernels.triton import (
    list_triton_kernels,
    TRITON_KERNEL_CATALOG,
    TRITON_STANDALONE_KERNELS,
    TRITON_CORE_MODULES,
    TRITON_ROOT_MODULES,
    TRITON_EXTENDED_KERNELS,
)
from neuralops.kernels.cuda import (
    list_cuda_kernels,
    CUDA_KERNEL_CATALOG,
)

# Common Hardware Dispatch & Diagnostic Flags
from neuralops.core.triton_kernels import (
    HAS_TRITON,
    FORCE_PYTORCH_FALLBACK,
    triton_dispatch,
    Autotuner,
    autotune_kernel,
    check_triton_suitability,
    log_fallback_reason,
)
from neuralops.core.cuda_kernels import (
    dispatch,
    force_kernel_backend,
    get_dispatch_stats,
    reset_dispatch_stats,
    is_cuda_available,
    get_device_capability,
    check_cuda_error,
    load_cuda_module,
    get_cuda_arch_flags,
)

# Standalone Triton Flagship Kernels
from neuralops.kernels.triton import (
    block_sparse_streaming_attn,
    chunked_delta_gate_scan,
    cosformer_linear_attention,
    fused_act_scale_quant_gemm,
    fused_hadamard_w4a16_gemm,
    fused_kronecker_matmul,
    fused_latent_kv_cache_attention,
    fused_lowrank_gemm,
    hgrn2_fused_bwd_scan,
    lut_codebook_dequant_gemm,
    rpca_sparse_gemm,
    spectral_entropy_svd_gate,
    strided_tt,
    triton_fused_lowrank_gated_gemm,
    triton_fused_lowrank_sparse_gemm,
    triton_strided_tt_contraction,
    triton_strided_tt_linear,
    triton_fused_hadamard_w4a16_gemm,
    triton_lut_codebook_dequant_gemm,
    triton_fused_kronecker_matmul,
    triton_chunked_delta_gate_scan,
    triton_cosformer_linear_attention,
    triton_block_sparse_streaming_attn,
    triton_hgrn2_fused_bwd_scan,
    triton_spectral_entropy_svd_gate,
    triton_fused_act_scale_quant_gemm,
    triton_fused_latent_kv_cache_attention,
)

# Core Triton Categorized Modules
from neuralops.core.triton_kernels import (
    linear_attn,
    sparse_attn,
    low_rank,
    state_update,
    gqa,
    fms_scan,
    gdn_scan,
    gla_scan,
    hgrn2_scan,
    selective_scan,
    selective_scan_fn,
    softplus,
    block_sparse,
    cosformer_triton_kernel,
    cur_linear,
    differential_attention,
    dynamic_sparse_attention,
    elementwise,
    flash_decoding,
    fused_ce,
    fused_kd,
    fused_norm_linear,
    fused_rope,
    paged_attention,
    retnet_retention,
    sliding_window,
    varlen_attention,
)
from neuralops.core.triton_linear_attn import fast_linear_attention, LinearAttentionTriton
from neuralops.core.triton_sparse_attn import triton_block_sparse_attention

# Extended Fused Kernels
from neuralops.kernels.extended_kernels import (
    fused_swiglu,
    fused_rmsnorm,
    fused_rotary_emb,
    fused_alibi_attention,
    fused_sink_token_attention,
    fused_decay_scan,
    fused_hadamard_transform,
    fused_int4_dequant,
    fused_fp8_e4m3_gemm,
    fused_spectral_dct,
    fused_chunked_cumsum,
    fused_cka_alignment,
    fused_mohawk_homotopy,
    fused_nullspace_project,
    fused_householder_reflect,
    fused_simplex_project,
    fused_topk_gating,
    fused_label_smoothing_ce,
    fused_kl_divergence,
    fused_lowrank_residual_add,
)

# Primary Native CUDA C++ Operators
from neuralops.core.cuda_kernels import (
    fused_lowrank_gated_gemm,
    hgrn2_fused_bwd_scan as cuda_hgrn2_fused_bwd_scan,
    cosformer_linear_attention as cuda_cosformer_linear_attention,
    fused_norm_linear as cuda_fused_norm_linear,
    fused_rotary_emb as cuda_fused_rotary_emb,
    kronecker_matmul as cuda_kronecker_matmul,
    lut_codebook_dequant_gemm as cuda_lut_codebook_dequant_gemm,
    linear_attention as cuda_linear_attention,
    fused_cross_entropy as cuda_fused_cross_entropy,
    rpca_sparse_svd_gemm as cuda_rpca_sparse_svd_gemm,
    tt_contraction as cuda_tt_contraction,
    tt_linear as cuda_tt_linear,
    hadamard_w4a16_gemm as cuda_hadamard_w4a16_gemm,
    act_scale_quant_gemm as cuda_act_scale_quant_gemm,
    latent_kv_cache_attention as cuda_latent_kv_cache_attention,
    spectral_entropy_svd_gate as cuda_spectral_entropy_svd_gate,
    fused_kd_loss as cuda_fused_kd_loss,
    recurrent_linear_step as cuda_recurrent_linear_step,
    flash_decoding_attention as cuda_flash_decoding_attention,
    fms_scan as cuda_fms_scan,
    rmsnorm as cuda_rmsnorm,
    layernorm as cuda_layernorm,
    swiglu as cuda_swiglu,
    fast_gelu as cuda_fast_gelu,
    fused_softplus as cuda_fused_softplus,
    gla_scan as cuda_gla_scan,
    causal_conv1d as cuda_causal_conv1d,
    recurrent_gla_step as cuda_recurrent_gla_step,
    recurrent_hgrn2_step as cuda_recurrent_hgrn2_step,
    fused_dropout_add as cuda_fused_dropout_add,
    varlen_attention as cuda_varlen_attention,
    cross_attention as cuda_cross_attention,
    rotational_svd as cuda_rotational_svd,
    block_diag_matmul as cuda_block_diag_matmul,
    ternary_gemm as cuda_ternary_gemm,
    fp4_gemm as cuda_fp4_gemm,
    channel_shuffle as cuda_channel_shuffle,
    speculative_verify as cuda_speculative_verify,
    alibi_attention as cuda_alibi_attention,
    flash_gqa as cuda_flash_gqa,
    moe_dispatch_gather as cuda_moe_dispatch_gather,
    paged_attention as cuda_paged_attention,
    copy_blocks_cow as cuda_copy_blocks_cow,
    w4a16_gemm as cuda_w4a16_gemm,
    fp8_gemm as cuda_fp8_gemm,
    selective_scan as cuda_selective_scan,
    sliding_window_attn as cuda_sliding_window_attn,
    block_sparse_attn as cuda_block_sparse_attn,
)

__all__ = [
    # Subpackages
    "triton",
    "cuda",
    # Discovery APIs
    "list_triton_kernels",
    "list_cuda_kernels",
    "TRITON_KERNEL_CATALOG",
    "TRITON_STANDALONE_KERNELS",
    "TRITON_CORE_MODULES",
    "TRITON_ROOT_MODULES",
    "TRITON_EXTENDED_KERNELS",
    "CUDA_KERNEL_CATALOG",
    # Dispatch & Diagnostics
    "HAS_TRITON",
    "FORCE_PYTORCH_FALLBACK",
    "triton_dispatch",
    "dispatch",
    "force_kernel_backend",
    "get_dispatch_stats",
    "reset_dispatch_stats",
    "is_cuda_available",
    "get_device_capability",
    "check_cuda_error",
    "load_cuda_module",
    "get_cuda_arch_flags",
    "Autotuner",
    "autotune_kernel",
    "check_triton_suitability",
    "log_fallback_reason",
    # Standalone Flagship Triton Modules
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
    # Standalone Triton Callables
    "triton_fused_lowrank_gated_gemm",
    "triton_fused_lowrank_sparse_gemm",
    "triton_strided_tt_contraction",
    "triton_strided_tt_linear",
    "triton_fused_hadamard_w4a16_gemm",
    "triton_lut_codebook_dequant_gemm",
    "triton_fused_kronecker_matmul",
    "triton_chunked_delta_gate_scan",
    "triton_cosformer_linear_attention",
    "triton_block_sparse_streaming_attn",
    "triton_hgrn2_fused_bwd_scan",
    "triton_spectral_entropy_svd_gate",
    "triton_fused_act_scale_quant_gemm",
    "triton_fused_latent_kv_cache_attention",
    # Core Triton Modules
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
    "selective_scan_fn",
    "sliding_window",
    "softplus",
    "sparse_attn",
    "state_update",
    "varlen_attention",
    # Root Bridge Modules
    "fast_linear_attention",
    "LinearAttentionTriton",
    "triton_block_sparse_attention",
    # Extended Fused Kernels
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
    # Primary CUDA Operators
    "fused_lowrank_gated_gemm",
    "cuda_hgrn2_fused_bwd_scan",
    "cuda_cosformer_linear_attention",
    "cuda_fused_norm_linear",
    "cuda_fused_rotary_emb",
    "cuda_kronecker_matmul",
    "cuda_lut_codebook_dequant_gemm",
    "cuda_linear_attention",
    "cuda_fused_cross_entropy",
    "cuda_rpca_sparse_svd_gemm",
    "cuda_tt_contraction",
    "cuda_tt_linear",
    "cuda_hadamard_w4a16_gemm",
    "cuda_act_scale_quant_gemm",
    "cuda_latent_kv_cache_attention",
    "cuda_spectral_entropy_svd_gate",
    "cuda_fused_kd_loss",
    "cuda_recurrent_linear_step",
    "cuda_flash_decoding_attention",
    "cuda_fms_scan",
    "cuda_rmsnorm",
    "cuda_layernorm",
    "cuda_swiglu",
    "cuda_fast_gelu",
    "cuda_fused_softplus",
    "cuda_gla_scan",
    "cuda_causal_conv1d",
    "cuda_recurrent_gla_step",
    "cuda_recurrent_hgrn2_step",
    "cuda_fused_dropout_add",
    "cuda_varlen_attention",
    "cuda_cross_attention",
    "cuda_rotational_svd",
    "cuda_block_diag_matmul",
    "cuda_ternary_gemm",
    "cuda_fp4_gemm",
    "cuda_channel_shuffle",
    "cuda_speculative_verify",
    "cuda_alibi_attention",
    "cuda_flash_gqa",
    "cuda_moe_dispatch_gather",
    "cuda_paged_attention",
    "cuda_copy_blocks_cow",
    "cuda_w4a16_gemm",
    "cuda_fp8_gemm",
    "cuda_selective_scan",
    "cuda_sliding_window_attn",
    "cuda_block_sparse_attn",
]


def __getattr__(name: str) -> Any:
    """Dynamically resolve attributes from triton or cuda subpackages."""
    if hasattr(triton, name):
        val = getattr(triton, name)
        globals()[name] = val
        return val
    if hasattr(cuda, name):
        val = getattr(cuda, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> list[str]:
    """Execute __dir__ operation."""
    return sorted(list(set(globals().keys()) | set(__all__)))
