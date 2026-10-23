#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
NeuralOps Triton Hardware Operators.
"""

from __future__ import annotations

from typing import Any

from . import (
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

from .fused_lowrank_gemm import (
    triton_fused_lowrank_gated_gemm,
    fused_lowrank_gemm,
)
from .rpca_sparse_gemm import triton_fused_lowrank_sparse_gemm
from .strided_tt import (
    triton_strided_tt_contraction,
    triton_strided_tt_linear,
)
from .fused_hadamard_w4a16_gemm import triton_fused_hadamard_w4a16_gemm
from .lut_codebook_dequant_gemm import triton_lut_codebook_dequant_gemm
from .fused_kronecker_matmul import triton_fused_kronecker_matmul
from .chunked_delta_gate_scan import triton_chunked_delta_gate_scan
from .cosformer_linear_attention import triton_cosformer_linear_attention
from .block_sparse_streaming_attn import triton_block_sparse_streaming_attn
from .hgrn2_fused_bwd_scan import triton_hgrn2_fused_bwd_scan
from .spectral_entropy_svd_gate import triton_spectral_entropy_svd_gate
from .fused_act_scale_quant_gemm import triton_fused_act_scale_quant_gemm
from .fused_latent_kv_cache_attention import (
    triton_fused_latent_kv_cache_attention,
)

from neuralops.ops.triton import (
    HAS_TRITON,
    FORCE_PYTORCH_FALLBACK,
    triton_dispatch,
    Autotuner,
    autotune_kernel,
    check_triton_suitability,
    log_fallback_reason,
    TRITON_STANDALONE_KERNELS,
    TRITON_CORE_MODULES,
    TRITON_ROOT_MODULES,
    TRITON_EXTENDED_KERNELS,
    TRITON_KERNEL_CATALOG,
    list_triton_kernels,
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

from neuralops.core.triton_linear_attn import fast_linear_attention, LinearAttentionTriton
from neuralops.core.triton_sparse_attn import triton_block_sparse_attention

__all__ = [
    "TRITON_STANDALONE_KERNELS",
    "TRITON_CORE_MODULES",
    "TRITON_ROOT_MODULES",
    "TRITON_EXTENDED_KERNELS",
    "TRITON_KERNEL_CATALOG",
    "list_triton_kernels",
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
    "HAS_TRITON",
    "FORCE_PYTORCH_FALLBACK",
    "triton_dispatch",
    "Autotuner",
    "autotune_kernel",
    "check_triton_suitability",
    "log_fallback_reason",
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
    "fast_linear_attention",
    "LinearAttentionTriton",
    "triton_block_sparse_attention",
]


def __getattr__(name: str) -> Any:
    """Raise for names not implemented or re-exported by this package."""
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> list[str]:
    """Execute __dir__ operation."""
    return sorted(list(set(globals().keys()) | set(__all__)))
