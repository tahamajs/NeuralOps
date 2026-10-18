"""
Aegis Gated Normalization Hardware Operators.

Author: Taha Majlesi
"""

from neuralops.core.cuda_kernels.aegis_gate_norm.aegis_gate_norm_autograd import (
    aegis_gate_norm,
    aegis_gate_norm_pytorch_ref,
)

__all__ = ["aegis_gate_norm", "aegis_gate_norm_pytorch_ref"]
