#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Low-Rank Bottleneck Projection and Modularized Submodules Facade.

Computes low-rank matrix bottleneck projections with fused bias addition,
quantization, autograd backward, and SVD compression routines.
Submodules are modularized under `neuralops.ops.triton._low_rank`.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

from neuralops.ops.triton._types import (
    FORCE_PYTORCH_FALLBACK,
    HAS_TRITON,
    check_triton_suitability,
    low_rank_pytorch_fallback,
    tl,
    triton,
)

from neuralops.ops.triton._low_rank.core import (
    _TritonLowRankAutogradFunction,
    _fused_low_rank_gemm_bwd_dx_kernel,
    _fused_low_rank_gemm_fwd_kernel,
    _fused_nystrom_pinv_step_fwd_kernel,
    _fused_primal_dual_landmark_fwd_kernel,
    _fused_rpc_sketch_attention_fwd_kernel,
    _fused_svd_mamba_proj_fwd_kernel,
    _triton_gemm_bias_kernel,
    bwd,
    fwd,
    is_supported,
    low_rank_forward,
)
from neuralops.ops.triton._low_rank.quantization import (
    BitPackedLowRankSVDLinear,
    BitPackedW4A16Linear,
    _dynamic_fp8_quant_kernel,
    _fp8_matmul_kernel,
    _w4a16_gemm_kernel,
    dynamic_fp8_quant_triton,
    fp8_tensorcore_matmul_triton,
    fused_w4a16_gemm_triton,
    pack_int4_weights,
    quantize_to_fp8_dynamic,
    unpack_int4_weights,
)
from neuralops.ops.triton._low_rank.fused_ops import (
    fused_galerkin_continuous_kernel_pytorch,
    fused_galerkin_continuous_kernel_triton,
    fused_int4_lut_dequant_matmul_pytorch,
    fused_linformer_seq_proj_pytorch,
    fused_linformer_seq_proj_triton,
    fused_lola_3way_decomposition_pytorch,
    fused_lola_3way_decomposition_triton,
    fused_nystrom_landmark_pinv_step_pytorch,
    fused_nystrom_landmark_pinv_step_triton,
    fused_primal_dual_landmark_pytorch,
    fused_primal_dual_landmark_triton,
    fused_rpc_sketch_attention_pytorch,
    fused_rpc_sketch_attention_triton,
    fused_svd_mamba_proj_pytorch,
    fused_svd_mamba_proj_triton,
)
from neuralops.ops.triton._low_rank.structured import (
    _fused_hnc_nullspace_spectral_step_fwd_kernel,
    _int4_lut_dequant_gemm_kernel,
    _kronecker_fused_gemm_kernel,
    _roast_hashed_gemm_kernel,
    fused_down_up_gemm_bias_triton,
    fused_int4_lut_dequant_matmul_triton,
    fused_kronecker_batched_gemm_pytorch,
    fused_kronecker_batched_gemm_triton,
    fused_residual_svd_gemm_triton,
    fused_roast_hashed_gather_gemm_pytorch,
    fused_roast_hashed_gather_gemm_triton,
    fused_salt_singular_mod_gemm_pytorch,
    fused_salt_singular_mod_gemm_triton,
    fused_svd_gemm_triton,
    fused_svd_lora_sparse_attn_triton,
    fused_svd_qkv_rope_attention_pytorch,
    fused_svd_qkv_rope_attention_triton,
    fused_tt_core_contraction_pytorch,
    fused_tt_core_contraction_triton,
)
from neuralops.ops.triton._low_rank.svd_steps import (
    _FusedHNCNullSpaceStepFn,
    block_diag_matmul_triton,
    fused_hnc_nullspace_step_pytorch,
    fused_hnc_nullspace_step_triton,
    rotational_svd_triton,
)
from neuralops.ops.triton._low_rank.kernels_gemm import (
    _fused_lowrank_gated_gemm_1d_kernel,
    _fused_rpca_gemm_kernel,
    fused_lowrank_gemm,
    triton_fused_lowrank_gated_gemm,
    triton_fused_lowrank_gated_gemm_pytorch_ref,
    triton_fused_lowrank_sparse_gemm,
    triton_fused_lowrank_sparse_gemm_pytorch_ref,
)
from neuralops.ops.triton._low_rank.kernels_tt_kron import (
    _fused_kronecker_gemm_kernel,
    _spectral_entropy_svd_proj_kernel,
    _strided_tt_reconstruct_kernel,
    triton_fused_kronecker_matmul,
    triton_fused_kronecker_matmul_pytorch_ref,
    triton_spectral_entropy_svd_gate,
    triton_spectral_entropy_svd_gate_pytorch_ref,
    triton_strided_tt_contraction,
    triton_strided_tt_contraction_pytorch_ref,
    triton_strided_tt_linear,
)
from neuralops.ops.triton._low_rank.svd_compress import (
    _asvd_reconstruct_kernel,
    _direct_svd_reconstruct_kernel,
    _fwsvd_reconstruct_kernel,
    _hnc_svd_reconstruct_kernel,
    asvd_compress,
    asvd_compress_pytorch,
    direct_svd_compress,
    direct_svd_compress_pytorch,
    fwsvd_compress,
    fwsvd_compress_pytorch,
    hnc_svd_compress,
    hnc_svd_compress_pytorch,
)

__all__ = [
    "is_supported",
    "fwd",
    "bwd",
    "low_rank_forward",
    "quantize_to_fp8_dynamic",
    "dynamic_fp8_quant_triton",
    "fp8_tensorcore_matmul_triton",
    "pack_int4_weights",
    "unpack_int4_weights",
    "fused_w4a16_gemm_triton",
    "BitPackedW4A16Linear",
    "BitPackedLowRankSVDLinear",
    "fused_linformer_seq_proj_triton",
    "fused_linformer_seq_proj_pytorch",
    "fused_galerkin_continuous_kernel_triton",
    "fused_galerkin_continuous_kernel_pytorch",
    "fused_nystrom_landmark_pinv_step_triton",
    "fused_nystrom_landmark_pinv_step_pytorch",
    "fused_rpc_sketch_attention_triton",
    "fused_rpc_sketch_attention_pytorch",
    "fused_primal_dual_landmark_triton",
    "fused_primal_dual_landmark_pytorch",
    "fused_svd_mamba_proj_triton",
    "fused_svd_mamba_proj_pytorch",
    "fused_lola_3way_decomposition_triton",
    "fused_lola_3way_decomposition_pytorch",
    "fused_int4_lut_dequant_matmul_triton",
    "fused_int4_lut_dequant_matmul_pytorch",
    "fused_roast_hashed_gather_gemm_triton",
    "fused_roast_hashed_gather_gemm_pytorch",
    "fused_kronecker_batched_gemm_triton",
    "fused_kronecker_batched_gemm_pytorch",
    "fused_tt_core_contraction_triton",
    "fused_tt_core_contraction_pytorch",
    "fused_salt_singular_mod_gemm_triton",
    "fused_salt_singular_mod_gemm_pytorch",
    "fused_svd_qkv_rope_attention_triton",
    "fused_svd_qkv_rope_attention_pytorch",
    "fused_svd_gemm_triton",
    "fused_down_up_gemm_bias_triton",
    "fused_residual_svd_gemm_triton",
    "fused_svd_lora_sparse_attn_triton",
    "fused_hnc_nullspace_step_triton",
    "fused_hnc_nullspace_step_pytorch",
    "block_diag_matmul_triton",
    "rotational_svd_triton",
    "triton_fused_lowrank_gated_gemm",
    "triton_fused_lowrank_gated_gemm_pytorch_ref",
    "triton_fused_lowrank_sparse_gemm",
    "triton_fused_lowrank_sparse_gemm_pytorch_ref",
    "triton_strided_tt_contraction",
    "triton_strided_tt_linear",
    "triton_strided_tt_contraction_pytorch_ref",
    "triton_fused_kronecker_matmul",
    "triton_fused_kronecker_matmul_pytorch_ref",
    "triton_spectral_entropy_svd_gate",
    "triton_spectral_entropy_svd_gate_pytorch_ref",
    "direct_svd_compress",
    "direct_svd_compress_pytorch",
    "asvd_compress",
    "asvd_compress_pytorch",
    "fwsvd_compress",
    "fwsvd_compress_pytorch",
    "hnc_svd_compress",
    "hnc_svd_compress_pytorch",
]
