#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
FP8 Quantized GEMM Autograd Interface

Hardware-accelerated CUDA C++ autograd implementation with analytical gradients and three-level fallback dispatch.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional
import torch
from neuralops.core.cuda_kernels._bindings import load_cuda_module
from neuralops.core.cuda_kernels._fallback import dispatch

_MODULE = None

def _get_module():
    global _MODULE
    if _MODULE is None:
        curr_dir = Path(__file__).parent
        with open(curr_dir / "fp8_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "fp8_gemm.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()

        _MODULE = load_cuda_module(
            name="cuda_fp8_gemm",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["fp8_gemm_fwd"],
        )
    return _MODULE

def fp8_gemm_pytorch_ref(
    A: torch.Tensor,
    B: torch.Tensor,
    scale_a: Optional[torch.Tensor] = None,
    scale_b: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    A_f = A.to(torch.float32)
    B_f = B.to(torch.float32)

    sa = scale_a.item() if scale_a is not None and scale_a.numel() > 0 else 1.0
    sb = scale_b.item() if scale_b is not None and scale_b.numel() > 0 else 1.0

    out = torch.matmul(A_f, B_f) * (sa * sb)
    if bias is not None and bias.numel() > 0:
        out = out + bias.to(torch.float32)
    return out.to(out_dtype)

def _get_triton_impl():
    return None

def fp8_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    scale_a: Optional[torch.Tensor] = None,
    scale_b: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    # The native kernel consumes packed uint8 FP8 payloads.  The public API
    # and existing model paths pass ordinary floating-point matrices; sending
    # those bytes to the FP8 decoder produces NaNs and invalid results.  Keep
    # the floating-point contract correct until an explicit packing API is
    # provided, while retaining the native path for already-packed tensors.
    if A.dtype != torch.uint8 or B.dtype != torch.uint8:
        return fp8_gemm_pytorch_ref(A, B, scale_a, scale_b, bias, out_dtype)

    def cuda_fn(A, B, scale_a=None, scale_b=None, bias=None, out_dtype=torch.float16, **kwargs):
        mod = _get_module()
        sa = scale_a if scale_a is not None else torch.empty(0, device=A.device)
        sb = scale_b if scale_b is not None else torch.empty(0, device=B.device)
        return mod.fp8_gemm_fwd(A.contiguous(), B.contiguous(), sa.contiguous(), sb.contiguous(), bias, out_dtype)

    triton_fn = _get_triton_impl()

    return dispatch(
        cuda_fn,
        triton_fn,
        fp8_gemm_pytorch_ref,
        A,
        B,
        scale_a=scale_a,
        scale_b=scale_b,
        bias=bias,
        out_dtype=out_dtype,
        kernel_name="fp8_gemm",
    )
