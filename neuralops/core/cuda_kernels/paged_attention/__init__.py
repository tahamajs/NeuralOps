"""
Paged Attention Hardware Operators.

Author: Taha Majlesi
"""

from neuralops.core.cuda_kernels.paged_attention.paged_attention_autograd import (
    paged_attention,
    paged_attention_pytorch_reference,
)

__all__ = ["paged_attention", "paged_attention_pytorch_reference"]
