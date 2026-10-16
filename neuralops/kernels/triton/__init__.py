"""
NeuralOps Triton Acceleration Kernels for SVD, RPCA, and Tensor-Train Compression.

Exposes high-performance fused kernels:
- triton_fused_lowrank_gated_gemm
- triton_fused_lowrank_sparse_gemm
- triton_strided_tt_contraction
- triton_strided_tt_linear
"""

from __future__ import annotations

from neuralops.kernels.triton.fused_lowrank_gemm import triton_fused_lowrank_gated_gemm
from neuralops.kernels.triton.rpca_sparse_gemm import triton_fused_lowrank_sparse_gemm
from neuralops.kernels.triton.strided_tt import (
    triton_strided_tt_contraction,
    triton_strided_tt_linear,
)

__all__ = [
    "triton_fused_lowrank_gated_gemm",
    "triton_fused_lowrank_sparse_gemm",
    "triton_strided_tt_contraction",
    "triton_strided_tt_linear",
]
