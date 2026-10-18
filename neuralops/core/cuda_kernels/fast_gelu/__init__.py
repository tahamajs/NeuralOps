"""
Fast GELU Activation Hardware Operators.

Author: Taha Majlesi
"""

from neuralops.core.cuda_kernels.fast_gelu.fast_gelu_autograd import (
    fast_gelu,
    fast_gelu_pytorch_ref,
)

__all__ = [
    "fast_gelu",
    "fast_gelu_pytorch_ref",
]
