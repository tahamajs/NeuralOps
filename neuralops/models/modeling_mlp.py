#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Multi-Layer Perceptron Architectures

Feed-forward neural network blocks supporting SwiGLU, GeLU, and standard gated MLP projections.
"""

from __future__ import annotations

import logging
import torch
import torch.nn as nn
from transformers.activations import ACT2FN

logger = logging.getLogger(__name__)


__all__ = ["NeuralOpsMLP", "NeuralOpsStandardMLP", "NeuralOpsSwiGLU", "build_mlp"]


class NeuralOpsSwiGLU(nn.Module):
    """SwiGLU / Gated MLP architecture as used in LLaMA / PaLM."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "silu",
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert callable(self.gate_proj), f"{type(self).__name__}.gate_proj is not callable"
        assert callable(self.up_proj), f"{type(self).__name__}.up_proj is not callable"
        assert callable(self.down_proj), f"{type(self).__name__}.down_proj is not callable"
        assert callable(self.act_fn), f"{type(self).__name__}.act_fn is not callable"
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        if x.is_cuda and x.dtype in (torch.float16, torch.bfloat16) and hasattr(self, "act_fn") and getattr(self.act_fn, "__name__", "") == "silu":
            try:
                from neuralops.ops.triton.elementwise import swiglu_triton
                act = swiglu_triton(gate, up)
                return self.down_proj(act)
            except (RuntimeError, ImportError, TypeError, ValueError, AttributeError) as e:
                logger.debug("swiglu_triton fallback to PyTorch: %s", e)
        return self.down_proj(self.act_fn(gate) * up)


class NeuralOpsStandardMLP(nn.Module):
    """Standard 2-layer MLP."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "gelu",
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.act_fn = ACT2FN[hidden_act]
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert callable(self.fc1), f"{type(self).__name__}.fc1 is not callable"
        assert callable(self.act_fn), f"{type(self).__name__}.act_fn is not callable"
        assert callable(self.fc2), f"{type(self).__name__}.fc2 is not callable"
        return self.fc2(self.act_fn(self.fc1(x)))


NeuralOpsMLP = NeuralOpsStandardMLP


def build_mlp(
    mlp_type: str,
    hidden_size: int,
    intermediate_size: int,
    hidden_act: str = "silu",
) -> nn.Module:
    """Build MLP layer according to configuration."""
    m_type = mlp_type.lower()
    if m_type in ("swiglu", "geglu", "gated"):
        return NeuralOpsSwiGLU(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
        )
    return NeuralOpsStandardMLP(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        hidden_act=hidden_act,
    )
