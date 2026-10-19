#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Causal 1D Convolution Autograd Interface

Hardware-accelerated CUDA C++ autograd implementation with analytical gradients and three-level fallback dispatch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from neuralops.core.cuda_kernels._bindings import load_cuda_module
from neuralops.core.cuda_kernels._fallback import dispatch

_MODULE = None


def _get_module():
    global _MODULE
    if _MODULE is None:
        curr_dir = Path(__file__).parent
        with open(curr_dir / "conv1d_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "causal_conv1d.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()

        _MODULE = load_cuda_module(
            name="cuda_causal_conv1d",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["causal_conv1d_fwd"],
        )
    return _MODULE


def causal_conv1d_pytorch_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    silu_activation: bool = False,
) -> torch.Tensor:
    """Pure PyTorch reference implementation."""
    B, D, L = x.shape
    if weight.dim() == 2:
        Kw = weight.shape[1]
        w = weight.unsqueeze(1)
    elif weight.dim() == 3:
        Kw = weight.shape[-1]
        w = weight
    else:
        raise ValueError(f"weight must be 2D or 3D, got shape {weight.shape}")
    x_pad = F.pad(x, (Kw - 1, 0))
    out = F.conv1d(x_pad, w, bias=bias, groups=D)
    if silu_activation:
        out = F.silu(out)
    return out


def _get_triton_impl():
    try:
        from neuralops.core.triton_kernels.state_update import causal_conv1d_step
        return causal_conv1d_step
    except (ImportError, AttributeError):
        return None


def causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    silu_activation: bool = False,
) -> torch.Tensor:
    """3-Level Dispatched Causal 1D Convolution."""
    def cuda_fn(x, weight, bias=None, silu_activation=False, **kwargs):
        mod = _get_module()
        return mod.causal_conv1d_fwd(
            x.contiguous(),
            weight.contiguous(),
            bias.contiguous() if bias is not None else None,
            silu_activation,
        )

    triton_fn = _get_triton_impl()

    return dispatch(
        cuda_fn,
        triton_fn,
        causal_conv1d_pytorch_ref,
        x,
        weight,
        bias=bias,
        silu_activation=silu_activation,
        kernel_name="causal_conv1d",
    )
