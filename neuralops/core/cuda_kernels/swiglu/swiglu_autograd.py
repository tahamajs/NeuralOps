# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Fused SwiGLU Activation Autograd Interface

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
        with open(curr_dir / "swiglu_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "swiglu.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()

        _MODULE = load_cuda_module(
            name="cuda_swiglu",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["swiglu_fwd"],
        )
    return _MODULE


def swiglu_pytorch_ref(
    x: torch.Tensor,
    gate: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Pure PyTorch reference implementation."""
    if gate is None:
        x_part, g_part = x.chunk(2, dim=-1)
    else:
        x_part, g_part = x, gate
    return F.silu(x_part) * g_part


def _get_triton_impl():
    try:
        from neuralops.core.triton_kernels.elementwise import swiglu_triton
        return swiglu_triton
    except (ImportError, AttributeError):
        return None


def swiglu(
    x: torch.Tensor,
    gate: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """3-Level Dispatched Fused SwiGLU."""
    def cuda_fn(x, gate=None, **kwargs):
        mod = _get_module()
        return mod.swiglu_fwd(
            x.contiguous(),
            gate.contiguous() if gate is not None else None,
        )

    triton_fn = _get_triton_impl()

    return dispatch(
        cuda_fn,
        triton_fn,
        swiglu_pytorch_ref,
        x,
        gate=gate,
        kernel_name="swiglu",
    )
