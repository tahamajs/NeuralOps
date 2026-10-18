"""
Fused Kronecker GEMM Hardware Operators.

Author: Taha Majlesi
"""

from neuralops.core.cuda_kernels.kronecker_gemm.kronecker_autograd import (
    kronecker_matmul,
    kronecker_matmul_pytorch_ref,
)

__all__ = [
    "kronecker_matmul",
    "kronecker_matmul_pytorch_ref",
]
