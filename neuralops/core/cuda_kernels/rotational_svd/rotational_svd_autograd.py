"""
Rotational Low-Rank SVD Interface and 3-Level Fallback.
Target: SM 8.9 (NVIDIA Ada Lovelace RTX 4070 Laptop GPU, 8GB VRAM).
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
        with open(curr_dir / "rotational_svd_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "rotational_svd.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()

        _MODULE = load_cuda_module(
            name="cuda_rotational_svd",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["rotational_svd_fwd"],
        )
    return _MODULE


def rotational_svd_pytorch_ref(
    x: torch.Tensor,
    r_in: torch.Tensor,
    v_r: torch.Tensor,
    u_r: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Pure PyTorch reference implementation."""
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    K = x_2d.shape[-1]
    V = v_r if v_r.shape[0] != K else v_r.t()
    U = u_r if u_r.shape[1] == V.shape[0] else u_r.t()
    N = U.shape[0]

    x_rot = torch.matmul(x_2d, r_in)
    z = torch.matmul(x_rot, V.t())
    y = torch.matmul(z, U.t())
    if bias is not None:
        y = y + bias
    return y.reshape(*orig_shape[:-1], N)


def _get_triton_impl():
    try:
        from neuralops.core.triton_kernels.low_rank import rotational_svd_triton
        return rotational_svd_triton
    except (ImportError, AttributeError):
        return None


def rotational_svd(
    x: torch.Tensor,
    r_in: torch.Tensor,
    v_r: torch.Tensor,
    u_r: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """3-Level Dispatched Rotational SVD."""
    def cuda_fn(x, r_in, v_r, u_r, bias=None, **kwargs):
        mod = _get_module()
        return mod.rotational_svd_fwd(
            x.contiguous(),
            r_in.contiguous(),
            v_r.contiguous(),
            u_r.contiguous(),
            bias.contiguous() if bias is not None else None,
        )

    triton_fn = _get_triton_impl()
    R = v_r.shape[1] if v_r.shape[0] == x.shape[-1] else v_r.shape[0]
    if R > 64:
        cuda_fn = None

    return dispatch(
        cuda_fn,
        triton_fn,
        rotational_svd_pytorch_ref,
        x,
        r_in,
        v_r,
        u_r,
        bias=bias,
        kernel_name="rotational_svd",
    )
