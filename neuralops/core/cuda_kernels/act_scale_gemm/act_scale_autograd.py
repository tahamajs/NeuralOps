#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Activation-Scaled Quantized GEMM Autograd Interface

Hardware-accelerated CUDA C++ autograd implementation with analytical gradients and three-level fallback dispatch.
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
        with open(curr_dir / "act_scale_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "act_scale_gemm.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()

        _MODULE = load_cuda_module(
            name="cuda_act_scale_gemm",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["act_scale_quant_gemm_fwd"],
        )
    return _MODULE


def act_scale_quant_gemm_pytorch_ref(
    x: torch.Tensor,
    s_inv: torch.Tensor,
    w: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Pure PyTorch reference implementation."""
    from neuralops.core.triton_kernels.fused_norm_linear import triton_fused_act_scale_quant_gemm_pytorch_ref
    return triton_fused_act_scale_quant_gemm_pytorch_ref(x, s_inv, w, bias=bias)


def _get_triton_impl():
    try:
        from neuralops.core.triton_kernels.fused_norm_linear import triton_fused_act_scale_quant_gemm
        return triton_fused_act_scale_quant_gemm
    except ImportError:
        return None


def _fwd_dispatch(x, s_inv, w, bias):
    def cuda_fn(x, s_inv, w, bias=None, **kwargs):
        mod = _get_module()
        return mod.act_scale_quant_gemm_fwd(
            x.contiguous(),
            s_inv.contiguous().to(x.dtype),
            w.contiguous(),
            bias.contiguous() if bias is not None else None,
        )

    triton_fn = _get_triton_impl()

    return dispatch(
        cuda_fn,
        triton_fn,
        act_scale_quant_gemm_pytorch_ref,
        x,
        s_inv,
        w,
        bias=bias,
        kernel_name="act_scale_quant_gemm",
    )


class _ActScaleQuantGemmFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        s_inv: torch.Tensor,
        w: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        orig_shape = x.shape
        K = orig_shape[-1]
        x_2d = x.reshape(-1, K)
        N = w.shape[0]

        out_2d = _fwd_dispatch(x_2d, s_inv, w, bias)

        ctx.save_for_backward(x_2d, s_inv, w)
        ctx.has_bias = bias is not None
        ctx.orig_shape = orig_shape
        ctx.out_features = N

        return out_2d.reshape(*orig_shape[:-1], N)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> tuple[torch.Tensor | None, ...]:
        dout = grad_outputs[0]
        x_2d, s_inv, w = ctx.saved_tensors
        dout_2d = dout.reshape(-1, ctx.out_features)

        s_inv_row = s_inv.view(1, -1)
        x_smooth = x_2d * s_inv_row

        dx_smooth = torch.matmul(dout_2d, w)

        dx_2d = dx_smooth * s_inv_row
        dx = dx_2d.reshape(ctx.orig_shape)

        ds_inv = None
        if ctx.needs_input_grad[1]:
            ds_inv = (dx_smooth * x_2d).sum(dim=0).to(s_inv.dtype)

        dw = None
        if ctx.needs_input_grad[2]:
            dw = torch.matmul(dout_2d.t(), x_smooth)

        dbias = None
        if ctx.has_bias and ctx.needs_input_grad[3]:
            dbias = dout_2d.sum(dim=0)

        return dx, ds_inv, dw, dbias


def act_scale_quant_gemm(
    x: torch.Tensor,
    s_inv: torch.Tensor,
    w: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """3-Level Dispatched Act-Scale GEMM with full autograd support."""
    needs_grad = torch.is_grad_enabled() and any(
        t.requires_grad for t in (x, s_inv, w, bias) if isinstance(t, torch.Tensor)
    )

    if needs_grad:
        return _ActScaleQuantGemmFunction.apply(x, s_inv, w, bias)

    return _fwd_dispatch(x, s_inv, w, bias)
