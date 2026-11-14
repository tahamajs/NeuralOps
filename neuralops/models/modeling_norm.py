#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Normalization Layer Implementations

Layer normalization and RMSNorm layers with optional fused Triton acceleration and scale calibration.
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
                try:
                    from neuralops.ops.triton.elementwise import fused_rmsnorm as _rmsnorm_fn
                except ImportError:
                    from neuralops.kernels.extended_kernels import rmsnorm_triton as _rmsnorm_fn

                weight = self.weight.to(device=hidden_states.device, dtype=hidden_states.dtype)
                out = _rmsnorm_fn(hidden_states, weight, self.variance_epsilon)
                return out.to(hidden_states.dtype)
            except (RuntimeError, ImportError) as e:
                logger.debug("Triton rmsnorm failed: %s, falling back to PyTorch reference", e)
        input_dtype = hidden_states.dtype
<<<<<<< HEAD
        hidden_states_f = hidden_states.to(torch.float32)
        variance = torch.rsqrt(hidden_states_f.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return ((hidden_states_f * variance).to(input_dtype)) * self.weight.to(input_dtype)
=======
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        weight = self.weight.to(device=hidden_states.device, dtype=input_dtype)
        return (weight * hidden_states.to(input_dtype)).to(input_dtype)
>>>>>>> fix/benchmarks-remediation


def build_norm_layer(norm_type: str, hidden_size: int, eps: float = 1e-6) -> nn.Module:
    """Build normalization layer based on configuration."""
    n_type = norm_type.lower()
    if n_type in ("rmsnorm", "rms_norm"):
        return NeuralOpsRMSNorm(hidden_size=hidden_size, eps=eps)
    elif n_type in ("layernorm", "layer_norm"):
        return nn.LayerNorm(hidden_size, eps=eps)
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}. Supported: 'rmsnorm', 'layernorm'")
