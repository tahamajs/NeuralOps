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
    # 10 Flagship CUDA C++ Packages
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
]
