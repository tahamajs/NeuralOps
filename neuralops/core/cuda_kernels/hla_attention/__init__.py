"""
Hadamard Linear Attention (HLA) Hardware Operators.

Author: Taha Majlesi
"""

from .hla_attention_autograd import hla_attention, hla_attention_ref

__all__ = ["hla_attention", "hla_attention_ref"]
