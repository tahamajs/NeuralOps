"""
NeuralOps Triton Hardware Operators.

Author: Taha Majlesi
"""

from __future__ import annotations

from typing import Any

# 1. Standalone Flagship Triton Kernel Modules
from neuralops.kernels.triton import (
    block_sparse_streaming_attn,
    chunked_delta_gate_scan,
    cosformer_linear_attention,
    fused_act_scale_quant_gemm,
    fused_hadamard_w4a16_gemm,
    fused_kronecker_matmul,
    fused_latent_kv_cache_attention,
    hgrn2_fused_bwd_scan,
    lut_codebook_dequant_gemm,
    rpca_sparse_gemm,
    spectral_entropy_svd_gate,
    strided_tt,
)

# 2. Flagship Kernel Functions
from neuralops.kernels.triton.fused_lowrank_gemm import (
    triton_fused_lowrank_gated_gemm,
    fused_lowrank_gemm,
)
from neuralops.kernels.triton.rpca_sparse_gemm import triton_fused_lowrank_sparse_gemm
from neuralops.kernels.triton.strided_tt import (
    triton_strided_tt_contraction,
    triton_strided_tt_linear,
)
from neuralops.kernels.triton.fused_hadamard_w4a16_gemm import triton_fused_hadamard_w4a16_gemm
from neuralops.kernels.triton.lut_codebook_dequant_gemm import triton_lut_codebook_dequant_gemm
from neuralops.kernels.triton.fused_kronecker_matmul import triton_fused_kronecker_matmul
from neuralops.kernels.triton.chunked_delta_gate_scan import triton_chunked_delta_gate_scan
from neuralops.kernels.triton.cosformer_linear_attention import triton_cosformer_linear_attention
from neuralops.kernels.triton.block_sparse_streaming_attn import triton_block_sparse_streaming_attn
from neuralops.kernels.triton.hgrn2_fused_bwd_scan import triton_hgrn2_fused_bwd_scan
from neuralops.kernels.triton.spectral_entropy_svd_gate import triton_spectral_entropy_svd_gate
from neuralops.kernels.triton.fused_act_scale_quant_gemm import triton_fused_act_scale_quant_gemm
from neuralops.kernels.triton.fused_latent_kv_cache_attention import (
    triton_fused_latent_kv_cache_attention,
)

# 3. Transparent Re-exports from Core Production Suite
from neuralops.core.triton_kernels import (
    # Hardware & Dispatch
    HAS_TRITON,
    FORCE_PYTORCH_FALLBACK,
    triton_dispatch,
    Autotuner,
    autotune_kernel,
    check_triton_suitability,
    log_fallback_reason,
    # Catalogs and Discovery
    TRITON_STANDALONE_KERNELS,
    TRITON_CORE_MODULES,
    TRITON_ROOT_MODULES,
    TRITON_EXTENDED_KERNELS,
    TRITON_KERNEL_CATALOG,
    list_triton_kernels,
    # PyTorch Reference Implementations for Flagship Kernels
    triton_fused_lowrank_gated_gemm_pytorch_ref,
    triton_fused_lowrank_sparse_gemm_pytorch_ref,
    triton_strided_tt_contraction_pytorch_ref,
    triton_fused_kronecker_matmul_pytorch_ref,
    triton_spectral_entropy_svd_gate_pytorch_ref,
    triton_fused_hadamard_w4a16_gemm_pytorch_ref,
    triton_lut_codebook_dequant_gemm_pytorch_ref,
    triton_fused_act_scale_quant_gemm_pytorch_ref,
    triton_chunked_delta_gate_scan_pytorch_ref,
    triton_cosformer_linear_attention_pytorch_ref,
    triton_block_sparse_streaming_attn_pytorch_ref,
    triton_hgrn2_fused_bwd_scan_pytorch_ref,
    triton_fused_latent_kv_cache_attention_pytorch_ref,
    # All Core Categorized Kernel Modules
    gdn_scan,
    gla_scan,
    hgrn2_scan,
    selective_scan,
    fms_scan,
    paged_attention,
    flash_decoding,
    sliding_window,
    fused_ce,
    fused_kd,
    fused_norm_linear,
    fused_rope,
    low_rank,
    elementwise,
    linear_attn,
    cosformer_triton_kernel,
    block_sparse,
    sparse_attn,
    gqa,
    varlen_attention,
    state_update,
    softplus,
    cur_linear,
    differential_attention,
    dynamic_sparse_attention,
    hydra_attention,
    retnet_retention,
    autotuner,
    hardware_profile,
    op_registry,
    _common,
    _types,
    # Extended Suite Functions
    triton_differential_attention,
    triton_differential_attention_pytorch_ref,
    TritonDifferentialAttentionFunction,
    triton_hydra_attention,
    triton_hydra_attention_pytorch_ref,
    TritonHydraAttentionFunction,
    triton_multi_scale_retention,
    triton_retnet_retention,
    triton_multi_scale_retention_pytorch_ref,
    triton_retnet_retention_pytorch_ref,
    TritonRetNetRetentionFunction,
    triton_dynamic_sparse_attention,
    triton_dynamic_sparse_attention_pytorch_ref,
    TritonDynamicSparseAttentionFunction,
    triton_cur_linear,
    triton_cur_linear_pytorch_ref,
    TritonCURLinearFunction,
)

# Root Bridge Modules
from neuralops.core.triton_linear_attn import fast_linear_attention, LinearAttentionTriton
from neuralops.core.triton_sparse_attn import triton_block_sparse_attention

__all__ = [
    # Catalog and discovery
    "TRITON_STANDALONE_KERNELS",
    "TRITON_CORE_MODULES",
    "TRITON_ROOT_MODULES",
    "TRITON_EXTENDED_KERNELS",
    "TRITON_KERNEL_CATALOG",
    "list_triton_kernels",
    # Flagship 13 Standalone Modules
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
    # Flagship 13 Kernel Callables
    "fused_lowrank_gemm",
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
    # PyTorch References
    "triton_fused_lowrank_gated_gemm_pytorch_ref",
    "triton_fused_lowrank_sparse_gemm_pytorch_ref",
    "triton_strided_tt_contraction_pytorch_ref",
    "triton_fused_kronecker_matmul_pytorch_ref",
    "triton_spectral_entropy_svd_gate_pytorch_ref",
    "triton_fused_hadamard_w4a16_gemm_pytorch_ref",
    "triton_lut_codebook_dequant_gemm_pytorch_ref",
    "triton_fused_act_scale_quant_gemm_pytorch_ref",
    "triton_chunked_delta_gate_scan_pytorch_ref",
    "triton_cosformer_linear_attention_pytorch_ref",
    "triton_block_sparse_streaming_attn_pytorch_ref",
    "triton_hgrn2_fused_bwd_scan_pytorch_ref",
    "triton_fused_latent_kv_cache_attention_pytorch_ref",
    # Dispatch & Diagnostics
    "HAS_TRITON",
    "FORCE_PYTORCH_FALLBACK",
    "triton_dispatch",
    "Autotuner",
    "autotune_kernel",
    "check_triton_suitability",
    "log_fallback_reason",
    # Core Categorized Modules
    "gdn_scan",
    "gla_scan",
    "hgrn2_scan",
    "selective_scan",
    "fms_scan",
    "paged_attention",
    "flash_decoding",
    "sliding_window",
    "fused_ce",
    "fused_kd",
    "fused_norm_linear",
    "fused_rope",
    "low_rank",
    "elementwise",
    "linear_attn",
    "cosformer_triton_kernel",
    "block_sparse",
    "sparse_attn",
    "gqa",
    "varlen_attention",
    "state_update",
    "softplus",
    "cur_linear",
    "differential_attention",
    "dynamic_sparse_attention",
    "hydra_attention",
    "retnet_retention",
    "autotuner",
    "hardware_profile",
    "op_registry",
    "_common",
    "_types",
    # Extended Suite Functions
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
    # Root Bridge Modules
    "fast_linear_attention",
    "LinearAttentionTriton",
    "triton_block_sparse_attention",
]


def __getattr__(name: str) -> Any:
    """Dynamically resolve attributes from neuralops.core.triton_kernels."""
    import neuralops.core.triton_kernels as ctk
    if hasattr(ctk, name):
        val = getattr(ctk, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> list[str]:
    """Execute __dir__ operation."""
    return sorted(list(set(globals().keys()) | set(__all__)))
