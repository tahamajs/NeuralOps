"""
Spectral Singular Value Shrinkage Hardware Operators.

Author: Taha Majlesi
"""

from neuralops.core.cuda_kernels.spectral_shrinkage.spectral_shrinkage_autograd import (
    spectral_shrinkage,
    spectral_shrinkage_pytorch_ref,
)

__all__ = ["spectral_shrinkage", "spectral_shrinkage_pytorch_ref"]
