"""NeuralOps CUDA C++ Kernel Package.

High-performance native CUDA C++ kernels augmenting and accelerating hot-paths
on modern NVIDIA architectures (Volta SM 7.0 through Blackwell SM 10.0).
Provides transparent 3-level fallback: CUDA C++ -> Triton JIT -> PyTorch Reference.
"""

from __future__ import annotations

from neuralops.core.cuda_kernels._bindings import load_cuda_module, get_cuda_arch_flags
from neuralops.core.cuda_kernels._fallback import (
    dispatch,
    force_kernel_backend,
    get_dispatch_stats,
    reset_dispatch_stats,
)
from neuralops.core.cuda_kernels._utils import (
    is_cuda_available,
    get_device_capability,
    check_cuda_error,
)

# 48 Native CUDA C++ Packages
from neuralops.core.cuda_kernels.low_rank_svd import (
    fused_lowrank_gated_gemm,
    fused_lowrank_gated_gemm_pytorch_ref,
)
from neuralops.core.cuda_kernels.gdn_scan import (
    chunked_delta_gate_scan,
    chunked_delta_gate_scan_pytorch_ref,
)
from neuralops.core.cuda_kernels.hgrn2_scan import (
    hgrn2_fused_bwd_scan,
    hgrn2_fused_bwd_scan_pytorch_ref,
)
from neuralops.core.cuda_kernels.cosformer import (
    cosformer_linear_attention,
    cosformer_linear_attention_pytorch_ref,
)
from neuralops.core.cuda_kernels.fused_norm_linear import (
    fused_norm_linear,
    fused_norm_linear_pytorch_ref,
)
from neuralops.core.cuda_kernels.fused_rope import (
    fused_rotary_emb,
    fused_rotary_emb_pytorch_ref,
)
from neuralops.core.cuda_kernels.kronecker_gemm import (
    kronecker_matmul,
    kronecker_matmul_pytorch_ref,
)
from neuralops.core.cuda_kernels.lut_gemm import (
    lut_codebook_dequant_gemm,
    lut_codebook_dequant_gemm_pytorch_ref,
)
from neuralops.core.cuda_kernels.linear_attention import (
    linear_attention,
    linear_attention_pytorch_ref,
)
from neuralops.core.cuda_kernels.fused_ce import (
    fused_cross_entropy,
    fused_cross_entropy_pytorch_ref,
)
from neuralops.core.cuda_kernels.rpca_sparse_svd import (
    rpca_sparse_svd_gemm,
    rpca_sparse_svd_gemm_pytorch_ref,
)
from neuralops.core.cuda_kernels.tt_contraction import (
    tt_contraction,
    tt_contraction_pytorch_ref,
    tt_linear,
    tt_linear_pytorch_ref,
)
from neuralops.core.cuda_kernels.hadamard_gemm import (
    hadamard_w4a16_gemm,
    hadamard_w4a16_gemm_pytorch_ref,
)
from neuralops.core.cuda_kernels.act_scale_gemm import (
    act_scale_quant_gemm,
    act_scale_quant_gemm_pytorch_ref,
)
from neuralops.core.cuda_kernels.latent_kv_attn import (
    latent_kv_cache_attention,
    latent_kv_cache_attention_pytorch_ref,
)
from neuralops.core.cuda_kernels.entropy_gate import (
    spectral_entropy_svd_gate,
    spectral_entropy_svd_gate_pytorch_ref,
)
from neuralops.core.cuda_kernels.fused_kd import (
    fused_kd_loss,
    fused_kd_loss_pytorch_ref,
)
from neuralops.core.cuda_kernels.state_update import (
    recurrent_linear_step,
    recurrent_linear_step_pytorch_ref,
)
from neuralops.core.cuda_kernels.flash_decoding import (
    flash_decoding_attention,
    flash_decoding_attention_pytorch_ref,
)
from neuralops.core.cuda_kernels.fms_scan import (
    fms_scan,
    fms_scan_pytorch_ref,
)
from neuralops.core.cuda_kernels.rmsnorm import (
    rmsnorm,
    rmsnorm_pytorch_ref,
)
from neuralops.core.cuda_kernels.layernorm import (
    layernorm,
    layernorm_pytorch_ref,
)
from neuralops.core.cuda_kernels.swiglu import (
    swiglu,
    swiglu_pytorch_ref,
)
from neuralops.core.cuda_kernels.fast_gelu import (
    fast_gelu,
    fast_gelu_pytorch_ref,
)
from neuralops.core.cuda_kernels.softplus import (
    fused_softplus,
    fused_softplus_pytorch_ref,
)
from neuralops.core.cuda_kernels.gla_scan import (
    gla_scan,
    gla_scan_pytorch_ref,
)
from neuralops.core.cuda_kernels.causal_conv1d import (
    causal_conv1d,
    causal_conv1d_pytorch_ref,
)
from neuralops.core.cuda_kernels.recurrent_gla_step import (
    recurrent_gla_step,
    recurrent_gla_step_pytorch_ref,
)
from neuralops.core.cuda_kernels.recurrent_hgrn2_step import (
    recurrent_hgrn2_step,
    recurrent_hgrn2_step_pytorch_ref,
)
from neuralops.core.cuda_kernels.fused_dropout_add import (
    fused_dropout_add,
    fused_dropout_add_pytorch_ref,
)
from neuralops.core.cuda_kernels.varlen_attention import (
    varlen_attention,
    varlen_attention_pytorch_ref,
)
from neuralops.core.cuda_kernels.cross_attention import (
    cross_attention,
    cross_attention_pytorch_ref,
)
from neuralops.core.cuda_kernels.rotational_svd import (
    rotational_svd,
    rotational_svd_pytorch_ref,
)
from neuralops.core.cuda_kernels.block_diag_gemm import (
    block_diag_matmul,
    block_diag_matmul_pytorch_ref,
)
from neuralops.core.cuda_kernels.ternary_gemm import (
    ternary_gemm,
    ternary_gemm_pytorch_ref,
)
from neuralops.core.cuda_kernels.fp4_gemm import (
    fp4_gemm,
    fp4_gemm_pytorch_ref,
)
from neuralops.core.cuda_kernels.channel_shuffle import (
    channel_shuffle,
    channel_shuffle_pytorch_ref,
)
from neuralops.core.cuda_kernels.speculative_verify import (
    speculative_verify,
    speculative_verify_pytorch_ref,
)
from neuralops.core.cuda_kernels.alibi_attention import (
    alibi_attention,
    alibi_attention_pytorch_ref,
)
from neuralops.core.cuda_kernels.multi_scale_retention import (
    multi_scale_retention,
    multi_scale_retention_pytorch_ref,
)

__all__ = [
    "load_cuda_module",
    "get_cuda_arch_flags",
    "dispatch",
    "force_kernel_backend",
    "get_dispatch_stats",
    "reset_dispatch_stats",
    "is_cuda_available",
    "get_device_capability",
    "check_cuda_error",
    # 48 Flagship CUDA C++ Operators
    "fused_lowrank_gated_gemm",
    "fused_lowrank_gated_gemm_pytorch_ref",
    "chunked_delta_gate_scan",
    "chunked_delta_gate_scan_pytorch_ref",
    "hgrn2_fused_bwd_scan",
    "hgrn2_fused_bwd_scan_pytorch_ref",
    "cosformer_linear_attention",
    "cosformer_linear_attention_pytorch_ref",
    "fused_norm_linear",
    "fused_norm_linear_pytorch_ref",
    "fused_rotary_emb",
    "fused_rotary_emb_pytorch_ref",
    "kronecker_matmul",
    "kronecker_matmul_pytorch_ref",
    "lut_codebook_dequant_gemm",
    "lut_codebook_dequant_gemm_pytorch_ref",
    "linear_attention",
    "linear_attention_pytorch_ref",
    "fused_cross_entropy",
    "fused_cross_entropy_pytorch_ref",
    "rpca_sparse_svd_gemm",
    "rpca_sparse_svd_gemm_pytorch_ref",
    "tt_contraction",
    "tt_contraction_pytorch_ref",
    "tt_linear",
    "tt_linear_pytorch_ref",
    "hadamard_w4a16_gemm",
    "hadamard_w4a16_gemm_pytorch_ref",
    "act_scale_quant_gemm",
    "act_scale_quant_gemm_pytorch_ref",
    "latent_kv_cache_attention",
    "latent_kv_cache_attention_pytorch_ref",
    "spectral_entropy_svd_gate",
    "spectral_entropy_svd_gate_pytorch_ref",
    "fused_kd_loss",
    "fused_kd_loss_pytorch_ref",
    "recurrent_linear_step",
    "recurrent_linear_step_pytorch_ref",
    "flash_decoding_attention",
    "flash_decoding_attention_pytorch_ref",
    "fms_scan",
    "fms_scan_pytorch_ref",
    "rmsnorm",
    "rmsnorm_pytorch_ref",
    "layernorm",
    "layernorm_pytorch_ref",
    "swiglu",
    "swiglu_pytorch_ref",
    "fast_gelu",
    "fast_gelu_pytorch_ref",
    "fused_softplus",
    "fused_softplus_pytorch_ref",
    "gla_scan",
    "gla_scan_pytorch_ref",
    "causal_conv1d",
    "causal_conv1d_pytorch_ref",
    "recurrent_gla_step",
    "recurrent_gla_step_pytorch_ref",
    "recurrent_hgrn2_step",
    "recurrent_hgrn2_step_pytorch_ref",
    "fused_dropout_add",
    "fused_dropout_add_pytorch_ref",
    "varlen_attention",
    "varlen_attention_pytorch_ref",
    "cross_attention",
    "cross_attention_pytorch_ref",
    "rotational_svd",
    "rotational_svd_pytorch_ref",
    "block_diag_matmul",
    "block_diag_matmul_pytorch_ref",
    "ternary_gemm",
    "ternary_gemm_pytorch_ref",
    "fp4_gemm",
    "fp4_gemm_pytorch_ref",
    "channel_shuffle",
    "channel_shuffle_pytorch_ref",
    "speculative_verify",
    "speculative_verify_pytorch_ref",
    "alibi_attention",
    "alibi_attention_pytorch_ref",
    "multi_scale_retention",
    "multi_scale_retention_pytorch_ref",
]
