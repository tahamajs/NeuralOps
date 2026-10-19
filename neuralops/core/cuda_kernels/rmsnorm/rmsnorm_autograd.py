# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Fused RMSNorm Autograd Interface

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
        with open(curr_dir / "rmsnorm_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "rmsnorm.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()

        _MODULE = load_cuda_module(
            name="cuda_rmsnorm",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["rmsnorm_fwd"],
        )
    return _MODULE


def rmsnorm_pytorch_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Pure PyTorch reference implementation."""
    orig_dtype = x.dtype
    x_eff = x.float() + (residual.float() if residual is not None else 0.0)
    rms = torch.rsqrt(x_eff.pow(2).mean(-1, keepdim=True) + eps)
    out = (x_eff * rms) * weight.float()
    return out.to(orig_dtype)


def _get_triton_impl():
    try:
        from neuralops.core.triton_kernels.elementwise import rmsnorm_triton
        return rmsnorm_triton
    except (ImportError, AttributeError):
        return None


def rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """3-Level Dispatched Fused RMSNorm."""
    def cuda_fn(x, weight, residual=None, eps=1e-6, **kwargs):
        mod = _get_module()
        return mod.rmsnorm_fwd(
            x.contiguous(),
            weight.contiguous(),
            residual.contiguous() if residual is not None else None,
            float(eps),
        )

    triton_fn = _get_triton_impl()

    return dispatch(
        cuda_fn,
        triton_fn,
        rmsnorm_pytorch_ref,
        x,
        weight,
        residual=residual,
        eps=eps,
        kernel_name="rmsnorm",
    )
