"""
NeuralOps Triton JIT GPU Acceleration Suite.

Canonical exports and seamless interoperability layer between:
  1. The 13 Standalone Flagship Triton JIT Kernels (SM 8.0 - SM 9.0+ Ada/Hopper):
     - triton_fused_lowrank_gated_gemm (fused_lowrank_gemm.py)
     - triton_fused_lowrank_sparse_gemm (rpca_sparse_gemm.py)
     - triton_strided_tt_contraction & triton_strided_tt_linear (strided_tt.py)
     - triton_fused_hadamard_w4a16_gemm (fused_hadamard_w4a16_gemm.py)
     - triton_lut_codebook_dequant_gemm (lut_codebook_dequant_gemm.py)
     - triton_fused_kronecker_matmul (fused_kronecker_matmul.py)
     - triton_chunked_delta_gate_scan (chunked_delta_gate_scan.py)
     - triton_cosformer_linear_attention (cosformer_linear_attention.py)
     - triton_block_sparse_streaming_attn (block_sparse_streaming_attn.py)
     - triton_hgrn2_fused_bwd_scan (hgrn2_fused_bwd_scan.py)
     - triton_spectral_entropy_svd_gate (spectral_entropy_svd_gate.py)
     - triton_fused_act_scale_quant_gemm (fused_act_scale_quant_gemm.py)
     - triton_fused_latent_kv_cache_attention (fused_latent_kv_cache_attention.py)
  2. The 26 Production Triton Kernel Modules & Autotuning Infrastructure
     hosted in `neuralops.core.triton_kernels`.
  3. PyTorch CPU / MPS / CUDA reference fallbacks for all kernels.
"""

from __future__ import annotations

import importlib
import sys
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
    fused_lowrank_gemm,
    hgrn2_fused_bwd_scan,
    lut_codebook_dequant_gemm,
    rpca_sparse_gemm,
    spectral_entropy_svd_gate,
    strided_tt,
)

# 2. Flagship Kernel Functions
from neuralops.kernels.triton.fused_lowrank_gemm import triton_fused_lowrank_gated_gemm
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
    # 26 Core Production Kernel Modules
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
    autotuner,
    hardware_profile,
    op_registry,
    _common,
    _types,
)

__all__ = [
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
    # 26 Core Modules
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
    "autotuner",
    "hardware_profile",
    "op_registry",
    "_common",
    "_types",
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
