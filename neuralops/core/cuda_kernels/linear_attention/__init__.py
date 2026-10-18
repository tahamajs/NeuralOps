"""
Causal Linear Attention Hardware Operators.

Author: Taha Majlesi
"""

from neuralops.core.cuda_kernels.linear_attention.linear_attn_autograd import (
    linear_attention,
    linear_attention_pytorch_ref,
)

__all__ = [
    "linear_attention",
    "linear_attention_pytorch_ref",
]
