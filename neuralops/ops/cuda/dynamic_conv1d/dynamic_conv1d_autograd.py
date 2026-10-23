#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Dynamic 1D Convolution Autograd Interface

Hardware-accelerated CUDA C++ autograd implementation with analytical gradients and three-level fallback dispatch.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional
import torch
import torch.nn.functional as F

from neuralops.ops.cuda._bindings import load_cuda_module
from neuralops.ops.cuda._fallback import dispatch

_MODULE = None

def _get_module():
    global _MODULE
    if _MODULE is None:
        curr_dir = Path(__file__).parent
        with open(curr_dir / "dynamic_conv1d_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "dynamic_conv1d.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()
        _MODULE = load_cuda_module(
            name="cuda_dynamic_conv1d",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["dynamic_conv1d_fwd"],
        )
    return _MODULE

from neuralops.ops.triton.elementwise import fused_dynamic_conv1d_pytorch, fused_dynamic_conv1d_triton

def dynamic_conv1d_pytorch_ref(v: torch.Tensor, weights: torch.Tensor, bias: Optional[torch.Tensor] = None, *args, **kwargs) -> torch.Tensor:
    if v.dim() == 3:
        B, D, L = v.shape
        K = weights.shape[1]
        padded = F.pad(v.float(), (K - 1, 0))
        w = weights.unsqueeze(1).float()
        b = bias.float() if bias is not None else None
        out = F.conv1d(padded, w, b, groups=D)
        return out.to(v.dtype)
    return fused_dynamic_conv1d_pytorch(v, weights, *args, **kwargs)

dynamic_conv1d_ref = dynamic_conv1d_pytorch_ref

def dynamic_conv1d(v: torch.Tensor, weights: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    def cuda_fn(*args, **kwargs):
        mod = _get_module()
        fn = mod.dynamic_conv1d_fwd
        return fn(*args, **kwargs)
    if weights.shape[-1] > 16 or (v.dim() == 4 and v.shape[-1] > 128):
        cuda_fn = None
    triton_fn = fused_dynamic_conv1d_triton if v.dim() == 4 else None
    return dispatch(
        cuda_fn,
        triton_fn,
        dynamic_conv1d_pytorch_ref,
        v, weights, bias,
        kernel_name="dynamic_conv1d",
    )
