"""Multi-Layer Perceptron (MLP) architectures for NeuralOps models."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers.activations import ACT2FN


__all__ = ["NeuralOpsMLP", "NeuralOpsSwiGLU"]


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
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


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
        return self.fc2(self.act_fn(self.fc1(x)))


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
