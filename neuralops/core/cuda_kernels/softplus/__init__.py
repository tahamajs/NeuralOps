"""
Fused Softplus Activation Hardware Operators.

Author: Taha Majlesi
"""

from neuralops.core.cuda_kernels.softplus.softplus_autograd import (
    fused_softplus,
    fused_softplus_pytorch_ref,
)

__all__ = [
    "fused_softplus",
    "fused_softplus_pytorch_ref",
]
