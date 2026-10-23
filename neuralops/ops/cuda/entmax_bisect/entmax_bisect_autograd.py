#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Entmax Bisection Autograd Interface

Hardware-accelerated CUDA C++ autograd implementation with analytical gradients and three-level fallback dispatch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F

from neuralops.ops.cuda._bindings import load_cuda_module
from neuralops.ops.cuda._fallback import dispatch

_MODULE = None


def _get_module():
    global _MODULE
    if _MODULE is None:
        curr_dir = Path(__file__).parent
        with open(curr_dir / "entmax_bisect_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "entmax_bisect.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()
        _MODULE = load_cuda_module(
            name="cuda_entmax_bisect",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["entmax15_fwd", "entmax15_bwd"],
        )
    return _MODULE


from neuralops.ops.triton.elementwise import fused_entmax_bisect_pytorch, fused_entmax_bisect_triton

entmax_bisect_pytorch_ref = fused_entmax_bisect_pytorch
entmax_bisect_ref = fused_entmax_bisect_pytorch


def _fwd_dispatch(scores, n_iter=15):
    def cuda_fn(scores, n_iter=15):
        mod = _get_module()
        return mod.entmax15_fwd(scores.contiguous(), int(n_iter))

    return dispatch(
        cuda_fn,
        fused_entmax_bisect_triton,
        fused_entmax_bisect_pytorch,
        scores,
        n_iter=n_iter,
        kernel_name="entmax_bisect",
    )


def entmax15_backward_cuda(dout: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    mod = _get_module()
    if mod is not None and hasattr(mod, "entmax15_bwd"):
        return mod.entmax15_bwd(dout.contiguous(), p.contiguous())
    p_f = p.float()
    dout_f = dout.float()
    sp = torch.sqrt(torch.clamp(p_f, min=0.0))
    mask = (p_f > 1e-9).float()
    sp = sp * mask
    sum_sp = sp.sum(dim=-1, keepdim=True).clamp(min=1e-9)
    sum_dout_sp = (dout_f * sp).sum(dim=-1, keepdim=True)
    ds = sp * (dout_f - (sum_dout_sp / sum_sp))
    return ds.to(dout.dtype)


class _Entmax15Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, scores: torch.Tensor, n_iter: int) -> torch.Tensor:
        p = _fwd_dispatch(scores, n_iter=n_iter)
        ctx.save_for_backward(p)
        return p

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> tuple[torch.Tensor | None, ...]:
        dout = grad_outputs[0]
        (p,) = ctx.saved_tensors
        if not ctx.needs_input_grad[0]:
            return None, None
        ds = entmax15_backward_cuda(dout, p)
        return ds, None


def entmax_bisect(scores: torch.Tensor, n_iter: int = 15) -> torch.Tensor:
    """3-Level Dispatched Entmax-1.5 Bisection with full autograd support."""
    if scores.requires_grad:
        return _Entmax15Function.apply(scores, n_iter)

    return _fwd_dispatch(scores, n_iter=n_iter)
