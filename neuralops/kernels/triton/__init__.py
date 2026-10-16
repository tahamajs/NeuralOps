"""
NeuralOps 13-Kernel Flagship Triton JIT GPU Acceleration Suite.

All 13 kernels targeting NVIDIA Ada Lovelace (RTX 4070, SM 8.9):
1. triton_fused_lowrank_gated_gemm: Fused low-rank GEMM with in-SRAM spectral gating & bias drift
2. triton_fused_lowrank_sparse_gemm: RPCA low-rank + sparse outlier epilogue in SRAM
3. triton_strided_tt_contraction & triton_strided_tt_linear: Zero-copy TT tensor contraction
4. triton_fused_hadamard_w4a16_gemm: FWHT rotation + register INT4 dequantization
5. triton_lut_codebook_dequant_gemm: In-register 16-entry centroid LUT lookup
6. triton_fused_kronecker_matmul: Zero-intermediate vec(B X A^T) Kronecker matmul
7. triton_chunked_delta_gate_scan: Chunked Gated Delta-Rule recurrent scan
8. triton_cosformer_linear_attention: On-the-fly sinusoidal modulation linear attention
9. triton_block_sparse_streaming_attn: Active-block table streaming FlashAttention
10. triton_hgrn2_fused_bwd_scan: Zero-flip bidirectional SSM recurrent scan
11. triton_spectral_entropy_svd_gate: Dynamic-rank spectral entropy masked low-rank GEMM
12. triton_fused_act_scale_quant_gemm: Fused activation smoothing SmoothQuant GEMM
13. triton_fused_latent_kv_cache_attention: Compressed rank-r latent KV-cache FlashAttention
"""

from __future__ import annotations

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

__all__ = [
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
]
