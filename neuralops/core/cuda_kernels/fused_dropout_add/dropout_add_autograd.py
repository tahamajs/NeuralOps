"""
Fused Dropout + Residual Addition Interface, Autograd Function, and 3-Level Fallback.
Target: SM 8.9 (NVIDIA Ada Lovelace RTX 4070 Laptop GPU, 8GB VRAM).
Forward: Y = dropout(x, p) + residual
Backward computes exact adjoint gradients for x and residual.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F

from neuralops.core.cuda_kernels._bindings import load_cuda_module
from neuralops.core.cuda_kernels._fallback import dispatch

_MODULE = None


def _get_module():
    global _MODULE
    if _MODULE is None:
        curr_dir = Path(__file__).parent
        with open(curr_dir / "dropout_add_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "fused_dropout_add.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()

        _MODULE = load_cuda_module(
            name="cuda_fused_dropout_add",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["fused_dropout_add_fwd"],
        )
    return _MODULE


def fused_dropout_add_pytorch_ref(
    x: torch.Tensor,
    residual: torch.Tensor,
    p: float = 0.0,
    training: bool = False,
    **kwargs: Any,
) -> torch.Tensor:
    """Pure PyTorch reference implementation."""
    dropped = F.dropout(x, p=p, training=training)
    return dropped + residual


def _get_triton_impl():
    try:
        from neuralops.core.triton_kernels.elementwise import dropout_add_triton
        return dropout_add_triton
    except (ImportError, AttributeError):
        return None


def _fwd_dispatch(x, residual, p, training, seed):
    def cuda_fn(x, residual, p=0.0, training=False, seed=0, **kwargs):
        mod = _get_module()
        return mod.fused_dropout_add_fwd(
            x.contiguous(),
            residual.contiguous(),
            float(p),
            bool(training),
            int(seed),
        )

    triton_fn = _get_triton_impl()

    return dispatch(
        cuda_fn,
        triton_fn,
        fused_dropout_add_pytorch_ref,
        x,
        residual,
        p=p,
        training=training,
        seed=seed,
        kernel_name=fused_dropout_add,
    )


class _FusedDropoutAddFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        residual: torch.Tensor,
        p: float,
        training: bool,
        seed: int,
    ) -> torch.Tensor:
        if training and p > 0.0:
            # Generate mask deterministically or via PyTorch generator
            # Keeping mask ensures exact adjoint backprop without discrepancy
            gen = torch.Generator(device=x.device).manual_seed(seed)
            mask = (torch.rand(x.shape, device=x.device, generator=gen) > p).to(x.dtype)
            scale = 1.0 / (1.0 - p)
            out = x * mask * scale + residual
            ctx.save_for_backward(mask)
            ctx.scale = scale
        else:
            out = x + residual
            ctx.save_for_backward(None)
            ctx.scale = 1.0

        ctx.training = training
        ctx.p = p
        return out

    @staticmethod
    def backward(ctx: Any, dout: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        (mask,) = ctx.saved_tensors
        dx = None
        if ctx.needs_input_grad[0]:
            if mask is not None:
                dx = dout * mask * ctx.scale
            else:
                dx = dout

        dresidual = None
        if ctx.needs_input_grad[1]:
            dresidual = dout

        return dx, dresidual, None, None, None


def fused_dropout_add(
    x: torch.Tensor,
    residual: torch.Tensor,
    p: float = 0.0,
    training: bool = False,
    seed: int | None = None,
) -> torch.Tensor:
    """3-Level Dispatched Fused Dropout + Add with full autograd support."""
    if seed is None:
        seed = int(torch.empty((), dtype=torch.int64).random_(0, 2**31).item())

    needs_grad = torch.is_grad_enabled() and any(
        t.requires_grad for t in (x, residual) if isinstance(t, torch.Tensor)
    )

    if needs_grad:
        return _FusedDropoutAddFunction.apply(x, residual, p, training, seed)

    return _fwd_dispatch(x, residual, p=p, training=training, seed=seed)
