#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Hadamard W4A16 Quantized GEMM Autograd Interface

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
        with open(curr_dir / "hadamard_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "hadamard_gemm.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()

        _MODULE = load_cuda_module(
            name="cuda_hadamard_gemm",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["hadamard_w4a16_gemm_fwd"],
        )
    return _MODULE


def hadamard_w4a16_gemm_pytorch_ref(
    x: torch.Tensor,
    w_packed: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    group_size: int = 32,
    bias: Optional[torch.Tensor] = None,
    apply_hadamard: bool = False,
) -> torch.Tensor:
    """Pure PyTorch reference implementation."""
    from neuralops.core.triton_kernels.fused_norm_linear import triton_fused_hadamard_w4a16_gemm_pytorch_ref
    return triton_fused_hadamard_w4a16_gemm_pytorch_ref(
        x, w_packed, scales, zeros, group_size=group_size, bias=bias, apply_hadamard=apply_hadamard
    )


def _get_triton_impl():
    try:
        from neuralops.core.triton_kernels.fused_norm_linear import triton_fused_hadamard_w4a16_gemm
        return triton_fused_hadamard_w4a16_gemm
    except ImportError:
        return None


def _fwd_dispatch(x, w_packed, scales, zeros, group_size=32, bias=None, apply_hadamard=False):
    def cuda_fn(x, w_packed, scales, zeros, group_size=32, bias=None, apply_hadamard=False, **kwargs):
        mod = _get_module()
        return mod.hadamard_w4a16_gemm_fwd(
            x.contiguous(),
            w_packed.contiguous(),
            scales.contiguous(),
            zeros.contiguous(),
            group_size,
            bias.contiguous() if bias is not None else None,
        )

    triton_fn = _get_triton_impl()

    return dispatch(
        cuda_fn,
        triton_fn,
        hadamard_w4a16_gemm_pytorch_ref,
        x,
        w_packed,
        scales,
        zeros,
        group_size=group_size,
        bias=bias,
        apply_hadamard=apply_hadamard,
        kernel_name="hadamard_w4a16_gemm",
    )


class _HadamardW4A16GemmFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        w_packed: torch.Tensor,
        scales: torch.Tensor,
        zeros: torch.Tensor,
        group_size: int,
        bias: torch.Tensor | None,
        apply_hadamard: bool,
    ) -> torch.Tensor:
        orig_shape = x.shape
        K = orig_shape[-1]
        x_2d = x.reshape(-1, K)
        N = w_packed.shape[0]

        out_2d = _fwd_dispatch(
            x_2d, w_packed, scales, zeros, group_size=group_size, bias=bias, apply_hadamard=apply_hadamard
        )

        ctx.save_for_backward(w_packed, scales, zeros)
        ctx.group_size = group_size
        ctx.has_bias = bias is not None
        ctx.apply_hadamard = apply_hadamard
        ctx.orig_shape = orig_shape
        ctx.out_features = N
        ctx.in_features = K
        ctx.x_dtype = x.dtype
        ctx.x_device = x.device

        return out_2d.reshape(*orig_shape[:-1], N)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> tuple[torch.Tensor | None, ...]:
        dout = grad_outputs[0]
        w_packed, scales, zeros = ctx.saved_tensors
        dout_2d = dout.reshape(-1, ctx.out_features)
        K = ctx.in_features
        group_size = ctx.group_size

        low = (w_packed & 0x0F).to(torch.int32)
        high = ((w_packed >> 4) & 0x0F).to(torch.int32)
        w_unpacked = torch.empty((ctx.out_features, K), dtype=torch.int32, device=w_packed.device)
        w_unpacked[:, 0::2] = low
        w_unpacked[:, 1::2] = high

        _num_groups = K // group_size
        scales_expanded = scales.repeat_interleave(group_size, dim=1)[:, :K].to(ctx.x_dtype)
        zeros_expanded = zeros.repeat_interleave(group_size, dim=1)[:, :K].to(ctx.x_dtype)
        w_dequant = (w_unpacked.to(ctx.x_dtype) - zeros_expanded) * scales_expanded

        dx_2d = torch.matmul(dout_2d, w_dequant)

        if ctx.apply_hadamard:
            from neuralops.core.triton_kernels.elementwise import fast_hadamard_transform
            dx_2d = fast_hadamard_transform(dx_2d)

        dx = dx_2d.reshape(ctx.orig_shape)

        dbias = None
        if ctx.has_bias and ctx.needs_input_grad[5]:
            dbias = dout_2d.sum(dim=0)

        return dx, None, None, None, None, dbias, None


def hadamard_w4a16_gemm(
    x: torch.Tensor,
    w_packed: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    group_size: int = 32,
    bias: Optional[torch.Tensor] = None,
    apply_hadamard: bool = False,
) -> torch.Tensor:
    """3-Level Dispatched Hadamard W4A16 GEMM with full autograd support."""
    needs_grad = torch.is_grad_enabled() and any(
        t.requires_grad for t in (x, bias) if isinstance(t, torch.Tensor)
    )

    if needs_grad:
        return _HadamardW4A16GemmFunction.apply(
            x, w_packed, scales, zeros, group_size, bias, apply_hadamard
        )

    return _fwd_dispatch(
        x, w_packed, scales, zeros, group_size=group_size, bias=bias, apply_hadamard=apply_hadamard
    )
