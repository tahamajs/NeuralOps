"""
Tensor-Train Strided Contraction Autograd Interface

Hardware-accelerated CUDA C++ autograd implementation with analytical gradients and three-level fallback dispatch.

Author: Taha Majlesi
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple

import torch

from neuralops.core.cuda_kernels._bindings import load_cuda_module
from neuralops.core.cuda_kernels._fallback import dispatch

_MODULE = None


def _get_module():
    global _MODULE
    if _MODULE is None:
        curr_dir = Path(__file__).parent
        with open(curr_dir / "tt_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "tt_contraction.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()

        _MODULE = load_cuda_module(
            name="cuda_tt_contraction",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["tt_contraction_fwd", "tt_linear_fwd"],
        )
    return _MODULE


def tt_contraction_pytorch_ref(
    core0: torch.Tensor,
    core1: torch.Tensor,
    p_dims: Tuple[int, int],
    q_dims: Tuple[int, int],
) -> torch.Tensor:
    """Pure PyTorch reference implementation."""
    from neuralops.core.triton_kernels.low_rank import triton_strided_tt_contraction_pytorch_ref
    return triton_strided_tt_contraction_pytorch_ref(core0, core1, p_dims, q_dims)


def tt_linear_pytorch_ref(
    x: torch.Tensor,
    core0: torch.Tensor,
    core1: torch.Tensor,
    p_dims: Tuple[int, int],
    q_dims: Tuple[int, int],
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Pure PyTorch reference linear implementation."""
    W = tt_contraction_pytorch_ref(core0, core1, p_dims, q_dims)
    out = torch.matmul(x, W.t().to(dtype=x.dtype))
    if bias is not None:
        out = out + bias.to(dtype=x.dtype)
    return out


def _get_triton_impl():
    try:
        from neuralops.core.triton_kernels.low_rank import triton_strided_tt_contraction
        return triton_strided_tt_contraction
    except ImportError:
        return None


def _get_triton_linear_impl():
    try:
        from neuralops.core.triton_kernels.low_rank import triton_strided_tt_linear
        return triton_strided_tt_linear
    except ImportError:
        return None


def tt_contraction(
    core0: torch.Tensor,
    core1: torch.Tensor,
    p_dims: Tuple[int, int],
    q_dims: Tuple[int, int],
) -> torch.Tensor:
    """3-Level Dispatched TT Contraction."""
    def cuda_fn(c0, c1, p_d, q_d, **kwargs):
        mod = _get_module()
        return mod.tt_contraction_fwd(c0.contiguous(), c1.contiguous(), p_d[0], p_d[1], q_d[0], q_d[1])

    triton_fn = _get_triton_impl()

    return dispatch(
        cuda_fn,
        triton_fn,
        tt_contraction_pytorch_ref,
        core0,
        core1,
        p_dims,
        q_dims,
        kernel_name="tt_contraction",
    )


def tt_linear(
    x: torch.Tensor,
    core0: torch.Tensor,
    core1: torch.Tensor,
    p_dims: Tuple[int, int],
    q_dims: Tuple[int, int],
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """3-Level Dispatched TT Linear."""
    def cuda_fn(x, c0, c1, p_d, q_d, bias=None, **kwargs):
        mod = _get_module()
        return mod.tt_linear_fwd(
            x.contiguous(),
            c0.contiguous(),
            c1.contiguous(),
            p_d[0],
            p_d[1],
            q_d[0],
            q_d[1],
            bias.contiguous() if bias is not None else None,
        )

    triton_fn = _get_triton_linear_impl()
    if core0.shape[-1] > 64:
        cuda_fn = None

    return dispatch(
        cuda_fn,
        triton_fn,
        tt_linear_pytorch_ref,
        x,
        core0,
        core1,
        p_dims,
        q_dims,
        bias=bias,
        kernel_name="tt_linear",
    )
