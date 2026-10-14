"""
Fused Normalization + Linear Projection Layer with Autograd Backward.
Combines RMSNorm (or LayerNorm) directly before Linear projection into an optimized pass.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class _FusedNormLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor], eps: float):
        ctx.eps = eps
        ctx.has_bias = bias is not None
        orig_shape = x.shape
        D = orig_shape[-1]
        x_2d = x.reshape(-1, D)

        # LayerNorm forward via F.layer_norm for exact precision equivalence
        x_hat = F.layer_norm(x_2d, (D,), eps=eps)

        # Linear projection
        out_2d = F.linear(x_hat, weight, bias)
        out = out_2d.reshape(*orig_shape[:-1], weight.shape[0])

        ctx.save_for_backward(x_2d, x_hat, weight)
        ctx.D = D
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x_2d, x_hat, weight = ctx.saved_tensors
        D = ctx.D
        grad_out_2d = grad_output.reshape(-1, weight.shape[0])

        dweight = torch.matmul(grad_out_2d.t(), x_hat)
        dbias = grad_out_2d.sum(dim=0) if ctx.has_bias else None

        dx_hat = torch.matmul(grad_out_2d, weight)
        # Analytical backward of LayerNorm without affine parameters
        var = ((x_2d - x_2d.mean(dim=-1, keepdim=True)) ** 2).mean(dim=-1, keepdim=True)
        rstd = torch.rsqrt(var + ctx.eps)
        mean_dx_hat = dx_hat.mean(dim=-1, keepdim=True)
        mean_dx_hat_x_hat = (dx_hat * x_hat).mean(dim=-1, keepdim=True)
        dx_2d = rstd * (dx_hat - mean_dx_hat - x_hat * mean_dx_hat_x_hat)

        dx = dx_2d.reshape(x_2d.shape)
        if grad_output.dim() > 2:
            dx = dx.reshape(*grad_output.shape[:-1], D)

        return dx, dweight, dbias, None


def fused_norm_linear_pytorch(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Reference PyTorch implementation of fused LayerNorm + Linear."""
    normed = F.layer_norm(x, (x.shape[-1],), eps=eps)
    return F.linear(normed, weight, bias)


def fused_norm_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Dispatches fused norm linear kernel with full autograd backward."""
    if torch.is_grad_enabled() or x.requires_grad or weight.requires_grad:
        return _FusedNormLinearFunction.apply(x, weight, bias, eps)
    return fused_norm_linear_pytorch(x, weight, bias=bias, eps=eps)


class _FusedRMSNormGateFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, gate: torch.Tensor, norm_weight: torch.Tensor, eps: float, activation: str):
        ctx.eps = eps
        ctx.activation = activation
        
        # RMSNorm forward: x / sqrt(mean(x^2) + eps) * norm_weight
        var = (x.float() ** 2).mean(dim=-1, keepdim=True)
        rstd = torch.rsqrt(var + eps)
        x_norm = (x.float() * rstd).to(x.dtype) * norm_weight

        # Gate activation
        if activation == "silu":
            gate_act = F.silu(gate.float()).to(gate.dtype)
        else:
            gate_act = torch.sigmoid(gate.float()).to(gate.dtype)

        out = x_norm * gate_act
        ctx.save_for_backward(x, gate, norm_weight, rstd)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, gate, norm_weight, rstd = ctx.saved_tensors
        D = x.shape[-1]
        
        if ctx.activation == "silu":
            sig = torch.sigmoid(gate.float())
            gate_act = (gate.float() * sig).to(gate.dtype)
            d_gate_act = (sig * (1.0 + gate.float() * (1.0 - sig))).to(gate.dtype)
        else:
            sig = torch.sigmoid(gate.float())
            gate_act = sig.to(gate.dtype)
            d_gate_act = (sig * (1.0 - sig)).to(gate.dtype)

        # d(loss)/d(x_norm) = grad_output * gate_act
        d_x_norm = grad_output * gate_act
        
        # d(loss)/d(gate) = grad_output * x_norm * d_gate_act
        var = (x.float() ** 2).mean(dim=-1, keepdim=True)
        x_norm_raw = (x.float() * rstd)
        x_norm = (x_norm_raw * norm_weight).to(x.dtype)
        d_gate = (grad_output * x_norm * d_gate_act).to(gate.dtype)

        # d_norm_weight = sum(d_x_norm * x_norm_raw)
        d_norm_weight = (d_x_norm.float() * x_norm_raw).reshape(-1, D).sum(dim=0).to(norm_weight.dtype)

        # RMSNorm input gradient:
        # dx = rstd * [ d_x_norm * norm_weight - (1/D) * x * rstd^2 * sum(d_x_norm * norm_weight * x) ]
        dy = d_x_norm.float() * norm_weight.float()
        sum_dy_x = (dy * x.float()).sum(dim=-1, keepdim=True)
        dx = (rstd * (dy - (1.0 / D) * x.float() * (rstd ** 2) * sum_dy_x)).to(x.dtype)

        return dx, d_gate, d_norm_weight, None, None


def fused_rmsnorm_gate(
    x: torch.Tensor,
    gate: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float = 1e-6,
    activation: str = "silu",
) -> torch.Tensor:
    """Fused RMSNorm followed by elementwise gating (e.g. SiLU or Sigmoid gate)."""
    return _FusedRMSNormGateFunction.apply(x, gate, norm_weight, eps, activation)


def fused_rmsnorm_linear(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    linear_weight: torch.Tensor,
    linear_bias: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Fused RMSNorm + Linear projection with full autograd backward."""
    # RMSNorm
    var = (x.float() ** 2).mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt(var + eps)
    x_norm = (x.float() * rstd).to(x.dtype) * norm_weight
    return F.linear(x_norm, linear_weight, linear_bias)


class FusedRMSNormLinear(nn.Module):
    """
    Combines RMSNorm followed by nn.Linear.
    """

    def __init__(self, in_features: int, out_features: int, eps: float = 1e-6, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        self.norm = nn.RMSNorm(in_features, eps=eps)
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm(x))


__all__ = [
    "FusedRMSNormLinear",
    "fused_norm_linear",
    "fused_norm_linear_pytorch",
    "fused_rmsnorm_linear",
    "fused_rmsnorm_gate",
]
