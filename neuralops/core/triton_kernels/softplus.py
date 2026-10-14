"""
Triton JIT and PyTorch Fallback implementation of Softplus activation with custom autograd backward.

Computes:
    softplus(x) = log(1 + exp(x))
    d/dx softplus(x) = sigmoid(beta * x)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from packaging import version

from neuralops.core.triton_kernels._common import HAS_TRITON, tl, triton

if HAS_TRITON and hasattr(triton, "__version__"):
    TRITON3 = version.parse(triton.__version__) >= version.parse("3.0.0")
else:
    TRITON3 = False


if HAS_TRITON and TRITON3:

    @triton.jit
    def softplus(dt):
        return tl.math.log(tl.math.exp(dt) + 1)

elif HAS_TRITON:

    @triton.jit
    def softplus(dt):
        return tl.math.log1p(tl.exp(dt))

else:
    def softplus(dt):
        return F.softplus(dt)


if HAS_TRITON and tl is not None:

    @triton.jit
    def _softplus_fwd_kernel(X_ptr, Y_ptr, N, BETA: tl.constexpr, THRESHOLD: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x = tl.load(X_ptr + offs, mask=mask, other=0.0)
        x_f = x.to(tl.float32)
        scaled_x = x_f * BETA
        # Numerical threshold check with FP32 math
        y_f = tl.where(scaled_x > THRESHOLD, x_f, tl.log(1.0 + tl.exp(scaled_x)) / BETA)
        tl.store(Y_ptr + offs, y_f.to(Y_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _softplus_bwd_kernel(X_ptr, DY_ptr, DX_ptr, N, BETA: tl.constexpr, THRESHOLD: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x = tl.load(X_ptr + offs, mask=mask, other=0.0)
        dy = tl.load(DY_ptr + offs, mask=mask, other=0.0)
        x_f = x.to(tl.float32)
        dy_f = dy.to(tl.float32)
        scaled_x = x_f * BETA
        # d/dx (log(1+exp(beta*x))/beta) = sigmoid(beta * x)
        sig = 1.0 / (1.0 + tl.exp(-scaled_x))
        dx_f = tl.where(scaled_x > THRESHOLD, dy_f, dy_f * sig)
        tl.store(DX_ptr + offs, dx_f.to(DX_ptr.dtype.element_ty), mask=mask)



class _SoftplusAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, beta: float, threshold: float):
        ctx.beta = float(beta)
        ctx.threshold = float(threshold)
        ctx.save_for_backward(x)

        if HAS_TRITON and x.is_cuda and x.is_contiguous():
            out = torch.empty_like(x)
            N = x.numel()
            grid = (triton.cdiv(N, 1024),)
            _softplus_fwd_kernel[grid](x, out, N, BETA=float(beta), THRESHOLD=float(threshold), BLOCK_SIZE=1024)
            return out
        return F.softplus(x, beta=beta, threshold=threshold)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        beta = ctx.beta
        threshold = ctx.threshold

        if HAS_TRITON and x.is_cuda and x.is_contiguous() and grad_output.is_contiguous():
            dx = torch.empty_like(x)
            N = x.numel()
            grid = (triton.cdiv(N, 1024),)
            _softplus_bwd_kernel[grid](x, grad_output, dx, N, BETA=beta, THRESHOLD=threshold, BLOCK_SIZE=1024)
            return dx, None, None

        scaled = x * beta
        sig = torch.sigmoid(scaled)
        dx = torch.where(scaled > threshold, grad_output, grad_output * sig)
        return dx, None, None


def _softplus_pytorch(x: torch.Tensor, beta: float = 1.0, threshold: float = 20.0) -> torch.Tensor:
    """Numerically stable PyTorch CPU/CUDA fallback for softplus."""
    return F.softplus(x, beta=beta, threshold=threshold)


def softplus_fn(x: torch.Tensor, beta: float = 1.0, threshold: float = 20.0) -> torch.Tensor:
    """Public dispatch function for softplus activation with autograd backward."""
    if torch.is_grad_enabled() or x.requires_grad:
        return _SoftplusAutograd.apply(x, beta, threshold)
    return _softplus_pytorch(x, beta=beta, threshold=threshold)


softplus_fwd = softplus_fn
softplus_pytorch = _softplus_pytorch

__all__ = ["softplus", "softplus_fn", "_softplus_pytorch", "softplus_fwd", "softplus_pytorch"]
