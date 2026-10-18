"""
Normalization Layer Implementations

Layer normalization and RMSNorm layers with optional fused Triton acceleration and scale calibration.

Author: Taha Majlesi
"""

from __future__ import annotations

import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


__all__ = ["NeuralOpsRMSNorm", "build_norm_layer"]


class NeuralOpsRMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (RMSNorm)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.is_cuda and hidden_states.dtype in (torch.float16, torch.bfloat16):
            try:
                from neuralops.core.triton_kernels.elementwise import rmsnorm_triton
                return rmsnorm_triton(hidden_states, self.weight, self.variance_epsilon)
            except (RuntimeError, ImportError) as e:
                logger.debug("rmsnorm_triton failed: %s, falling back to PyTorch reference", e)
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def build_norm_layer(norm_type: str, hidden_size: int, eps: float = 1e-6) -> nn.Module:
    """Build normalization layer based on configuration."""
    n_type = norm_type.lower()
    if n_type in ("rmsnorm", "rms_norm"):
        return NeuralOpsRMSNorm(hidden_size=hidden_size, eps=eps)
    elif n_type in ("layernorm", "layer_norm"):
        return nn.LayerNorm(hidden_size, eps=eps)
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}. Supported: 'rmsnorm', 'layernorm'")
