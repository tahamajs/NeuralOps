"""
Fused SwiGLU Activation Hardware Operators.

Author: Taha Majlesi
"""

from neuralops.core.cuda_kernels.swiglu.swiglu_autograd import (
    swiglu,
    swiglu_pytorch_ref,
)

__all__ = [
    "swiglu",
    "swiglu_pytorch_ref",
]
