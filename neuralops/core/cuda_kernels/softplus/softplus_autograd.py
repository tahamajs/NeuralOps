# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Fused Softplus Activation Autograd Interface

Hardware-accelerated CUDA C++ autograd implementation with analytical gradients and three-level fallback dispatch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from neuralops.core.cuda_kernels._bindings import load_cuda_module
from neuralops.core.cuda_kernels._fallback import dispatch

_MODULE = None


def _get_module():
    global _MODULE
    if _MODULE is None:
        curr_dir = Path(__file__).parent
        with open(curr_dir / "softplus_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "softplus.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()

        _MODULE = load_cuda_module(
            name="cuda_softplus",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["softplus_fwd", "softplus_bwd"],
        )
    return _MODULE


def fused_softplus_pytorch_ref(
    x: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> torch.Tensor:
    """Pure PyTorch reference implementation."""
    return F.softplus(x, beta=beta, threshold=threshold)


def _get_triton_impl():
    try:
        from neuralops.core.triton_kernels.softplus import softplus_triton
        return softplus_triton
    except (ImportError, AttributeError):
        return None


class _FusedSoftplusFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        beta: float,
        threshold: float,
    ) -> torch.Tensor:
        mod = _get_module()
        out = mod.softplus_fwd(x.contiguous(), float(beta), float(threshold))
        ctx.save_for_backward(x)
        ctx.beta = float(beta)
        ctx.threshold = float(threshold)
        return out

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> tuple[torch.Tensor | None, ...]:
        dout = grad_outputs[0]
        (x,) = ctx.saved_tensors
        mod = _get_module()
        dx = mod.softplus_bwd(dout.contiguous(), x.contiguous(), ctx.beta, ctx.threshold)
        return dx, None, None


def fused_softplus(
    x: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> torch.Tensor:
    """3-Level Dispatched Fused Softplus with Full Autograd Support."""
    needs_grad = torch.is_grad_enabled() and isinstance(x, torch.Tensor) and x.requires_grad

    if x.is_cuda and needs_grad:
        return _FusedSoftplusFunction.apply(x, float(beta), float(threshold))

    def cuda_fn(x, beta=1.0, threshold=20.0, **kwargs):
        if needs_grad:
            return _FusedSoftplusFunction.apply(x, float(beta), float(threshold))
        mod = _get_module()
        return mod.softplus_fwd(x.contiguous(), float(beta), float(threshold))

    triton_fn = _get_triton_impl()

    return dispatch(
        cuda_fn,
        triton_fn,
        fused_softplus_pytorch_ref,
        x,
        beta=beta,
        threshold=threshold,
        kernel_name="fused_softplus",
    )
