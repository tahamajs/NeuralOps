"""
Rotational Low-Rank SVD Interface, Autograd Function, and 3-Level Fallback.
Target: SM 8.9 (NVIDIA Ada Lovelace RTX 4070 Laptop GPU, 8GB VRAM).
Forward: Y = (((X @ R_in) @ V_r.T) @ U_r.T) + bias
Backward computes exact analytical adjoint gradients for X, R_in, V_r, U_r, bias.
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
        kernel_name=rotational_svd,
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

        out_2d = _fwd_dispatch(x_2d, r_in, V, U, bias)

        ctx.save_for_backward(x_2d, r_in, V, U)
        ctx.has_bias = bias is not None
        ctx.orig_shape = orig_shape
        ctx.out_features = N
        ctx.v_r_transposed = (v_r.shape[0] == K)
        ctx.u_r_transposed = (u_r.shape[1] != R)

        return out_2d.reshape(*orig_shape[:-1], N)

    @staticmethod
    def backward(ctx: Any, dout: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        x_2d, r_in, V, U = ctx.saved_tensors
        dout_2d = dout.reshape(-1, ctx.out_features)

        # Adjoint backward:
        # x_rot = x_2d @ r_in
        # z = x_rot @ V.T
        # y = z @ U.T + bias
        x_rot = torch.matmul(x_2d, r_in)
        z = torch.matmul(x_rot, V.t())

        # dz = dout_2d @ U  [M, R]
        dz = torch.matmul(dout_2d, U)

        # dx_rot = dz @ V  [M, K]
        dx_rot = torch.matmul(dz, V)

        # dx = dx_rot @ r_in.T  [M, K]
        dx_2d = torch.matmul(dx_rot, r_in.t())
        dx = dx_2d.reshape(ctx.orig_shape)

        dr_in = None
        if ctx.needs_input_grad[1]:
            dr_in = torch.matmul(x_2d.t(), dx_rot)

        dV = None
        if ctx.needs_input_grad[2]:
            dV = torch.matmul(dz.t(), x_rot)
            if ctx.v_r_transposed:
                dV = dV.t()

        dU = None
        if ctx.needs_input_grad[3]:
            dU = torch.matmul(dout_2d.t(), z)
            if ctx.u_r_transposed:
                dU = dU.t()

        dbias = None
        if ctx.has_bias and ctx.needs_input_grad[4]:
            dbias = dout_2d.sum(dim=0)

        return dx, dr_in, dV, dU, dbias


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
