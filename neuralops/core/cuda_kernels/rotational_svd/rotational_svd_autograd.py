"""
Rotational Low-Rank SVD Autograd Interface

Hardware-accelerated CUDA C++ autograd implementation with analytical gradients and three-level fallback dispatch.

Author: Taha Majlesi
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
        with open(curr_dir / "rotational_svd_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "rotational_svd.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()

        _MODULE = load_cuda_module(
            name="cuda_rotational_svd",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["rotational_svd_fwd", "rotational_svd_bwd"],
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


def _fwd_dispatch(x, r_in, v_r, u_r, bias):
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


class _RotationalSVDFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        r_in: torch.Tensor,
        v_r: torch.Tensor,
        u_r: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        orig_shape = x.shape
        K = orig_shape[-1]
        x_2d = x.reshape(-1, K)

        V = v_r if v_r.shape[0] != K else v_r.t()
        R = V.shape[0]
        U = u_r if u_r.shape[1] == R else u_r.t()
        N = U.shape[0]

        mod = _get_module()
        x_rot = torch.matmul(x_2d, r_in)
        z = torch.matmul(x_rot, V.t())
        out_2d = torch.matmul(z, U.t())
        if bias is not None:
            out_2d = out_2d + bias

        ctx.save_for_backward(x_2d, r_in, v_r, u_r, x_rot, z)
        ctx.has_bias = bias is not None
        ctx.orig_shape = orig_shape
        ctx.out_features = N

        return out_2d.reshape(*orig_shape[:-1], N)

    @staticmethod
    def backward(ctx: Any, dout: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        x_2d, r_in, v_r, u_r, x_rot, z = ctx.saved_tensors
        dout_2d = dout.reshape(-1, ctx.out_features).contiguous()

        mod = _get_module()
        grads = mod.rotational_svd_bwd(dout_2d, x_2d, r_in, v_r, u_r, x_rot, z)
        dx, dr_in, dV, dU = grads[0], grads[1], grads[2], grads[3]

        dbias = None
        if ctx.has_bias:
            dbias = dout_2d.sum(dim=0)

        return dx.reshape(ctx.orig_shape), dr_in, dV, dU, dbias


def rotational_svd(
    x: torch.Tensor,
    r_in: torch.Tensor,
    v_r: torch.Tensor,
    u_r: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """3-Level Dispatched Rotational SVD with full autograd support."""
    needs_grad = torch.is_grad_enabled() and any(
        t.requires_grad for t in (x, r_in, v_r, u_r, bias) if isinstance(t, torch.Tensor)
    )

    if needs_grad:
        return _RotationalSVDFunction.apply(x, r_in, v_r, u_r, bias)

    return _fwd_dispatch(x, r_in, v_r, u_r, bias)
