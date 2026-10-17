"""
Phantom Gated Sparsity Interface, Autograd Function, and 3-Level Fallback.
Target: SM 8.9 (NVIDIA Ada Lovelace RTX 4070 Laptop GPU, 8GB VRAM).
Forward: g = sigmoid(gate)
         mask = where(g >= threshold, g, 0)
         out = x * mask
Backward computes exact analytical adjoint gradients for x and gate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch

from neuralops.core.cuda_kernels._bindings import load_cuda_module
from neuralops.core.cuda_kernels._fallback import dispatch

_MODULE = None


def _get_module():
    global _MODULE
    if _MODULE is None:
        curr_dir = Path(__file__).parent
        with open(curr_dir / "phantom_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "phantom.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()
        _MODULE = load_cuda_module("cuda_phantom", cpp_src, cuda_src, ["phantom_fwd"])
    return _MODULE


def phantom_ref(x: torch.Tensor, gate: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Pure PyTorch reference implementation with FP32 precision for threshold boundary."""
    g = torch.sigmoid(gate.float())
    mask = torch.where(g >= threshold, g, torch.zeros_like(g)).to(x.dtype)
    return x * mask


def _fwd_dispatch(x, gate, threshold=0.5):
    def cuda_fn(x, gate, threshold=0.5, **kwargs):
        mod = _get_module()
        return mod.phantom_fwd(x.contiguous(), gate.contiguous(), float(threshold))

    return dispatch(cuda_fn, phantom_ref, phantom_ref, x, gate, threshold=threshold, kernel_name=phantom)


class _PhantomFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        gate: torch.Tensor,
        threshold: float,
    ) -> torch.Tensor:
        out = _fwd_dispatch(x, gate, threshold=threshold)
        ctx.save_for_backward(x, gate)
        ctx.threshold = threshold
        return out

    @staticmethod
    def backward(ctx: Any, dout: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        x, gate = ctx.saved_tensors
        threshold = ctx.threshold

        g = torch.sigmoid(gate.float())
        active = (g >= threshold)
        mask = torch.where(active, g, torch.zeros_like(g)).to(x.dtype)

        dx = None
        if ctx.needs_input_grad[0]:
            dx = dout * mask

        dgate = None
        if ctx.needs_input_grad[1]:
            # For g >= threshold: d/dgate [x * g] = x * g * (1 - g)
            dg = g * (1.0 - g)
            dgate = (dout.float() * x.float() * dg * active.float()).to(gate.dtype)

        return dx, dgate, None


def phantom(x: torch.Tensor, gate: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """3-Level Dispatched Phantom Gated Sparsity with full autograd support."""
    needs_grad = torch.is_grad_enabled() and any(
        t.requires_grad for t in (x, gate) if isinstance(t, torch.Tensor)
    )

    if needs_grad:
        return _PhantomFunction.apply(x, gate, threshold)

    return _fwd_dispatch(x, gate, threshold=threshold)
