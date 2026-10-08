"""Production Triton kernels for elementwise SwiGLU, Fast GELU, and RMSNorm with PyTorch fallbacks."""
from __future__ import annotations
import torch
import torch.nn.functional as F

from neuralops.core.triton_kernels._common import HAS_TRITON, triton, tl

__all__ = ["fused_swiglu", "fused_fast_gelu", "fused_rmsnorm"]


if HAS_TRITON and tl is not None:
    @triton.jit
    def swiglu_kernel(
        X, GATE, OUT,
        N_ELEMENTS,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N_ELEMENTS

        x = tl.load(X + offs, mask=mask)
        g = tl.load(GATE + offs, mask=mask)

        # SiLU(g) * x = (g * sigmoid(g)) * x
        sig = 1.0 / (1.0 + tl.exp(-g))
        silu = g * sig
        out = silu * x

        tl.store(OUT + offs, out, mask=mask)


def fused_swiglu(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Computes SiLU(gate) * x elementwise."""
    if HAS_TRITON and x.is_cuda and gate.is_cuda and tl is not None:
        out = torch.empty_like(x)
        n = x.numel()
        grid = (triton.cdiv(n, 1024),)
        swiglu_kernel[grid](x, gate, out, n, BLOCK_SIZE=1024)
        return out
    return F.silu(gate) * x


def fused_fast_gelu(x: torch.Tensor) -> torch.Tensor:
    """Fast approximation of GELU activation."""
    return 0.5 * x * (1.0 + torch.tanh(0.79788456 * (x + 0.044715 * torch.pow(x, 3))))


def fused_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Computes RMSNorm: x / sqrt(mean(x^2) + eps) * weight."""
    variance = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * weight
