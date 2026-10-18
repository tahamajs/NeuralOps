"""
FAVOR+ Random Feature Attention Hardware Operators.

Author: Taha Majlesi
"""

from neuralops.core.cuda_kernels.favor_plus.favor_plus_autograd import (
    favor_plus,
    favor_plus_pytorch_ref,
)

__all__ = ["favor_plus", "favor_plus_pytorch_ref"]
