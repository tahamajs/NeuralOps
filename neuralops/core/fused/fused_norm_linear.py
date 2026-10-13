"""
Fused normalization + linear layers.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNormLinear(nn.Module):
    """Fused RMSNorm followed by Linear."""

    def __init__(
        self,
        d_model: int,
        out_features: int,
        eps: float = 1e-6,
        bias: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.out_features = out_features
        self.eps = eps

        self.norm_weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.linear = nn.Linear(d_model, out_features, bias=bias, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x_norm = x * torch.rsqrt(variance + self.eps) * self.norm_weight
        return self.linear(x_norm)


class LayerNormLinear(nn.Module):
    """Fused LayerNorm + Linear."""

    def __init__(
        self,
        d_model: int,
        out_features: int,
        eps: float = 1e-5,
        bias: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, eps=eps, device=device, dtype=dtype)
        self.linear = nn.Linear(d_model, out_features, bias=bias, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm(x))


class RMSNormGated(nn.Module):
    """RMSNorm with element-wise gating."""

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-6,
        gating_type: str = "sigmoid",
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.gating_type = gating_type

        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.gate_proj = nn.Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x_norm = x * torch.rsqrt(variance + self.eps) * self.weight
        gate = self.gate_proj(x)
        if self.gating_type == "sigmoid":
            gate = torch.sigmoid(gate)
        elif self.gating_type == "swish":
            gate = F.silu(gate)
        else:
            raise ValueError(f"Unknown gating: {self.gating_type}")
        return x_norm * gate


class LayerNormGated(nn.Module):
    """LayerNorm with element-wise gating."""

    def __init__(self, d_model: int, eps: float = 1e-5, gating_type: str = "sigmoid"):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, eps=eps)
        self.gate_proj = nn.Linear(d_model, d_model)
        self.gating_type = gating_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm(x)
        gate = self.gate_proj(x)
        if self.gating_type == "sigmoid":
            gate = torch.sigmoid(gate)
        elif self.gating_type == "swish":
            gate = F.silu(gate)
        return x_norm * gate
