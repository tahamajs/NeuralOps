"""
Fused Cross-Entropy Loss Hardware Operators.

Author: Taha Majlesi
"""

from neuralops.core.cuda_kernels.fused_ce.fused_ce_autograd import (
    fused_cross_entropy,
    fused_cross_entropy_pytorch_ref,
)

__all__ = [
    "fused_cross_entropy",
    "fused_cross_entropy_pytorch_ref",
]
