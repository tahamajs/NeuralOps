"""
RPCA Low-Rank + Sparse SVD GEMM Interface, Autograd Function, and 3-Level Fallback.
Target: SM 8.9 (NVIDIA Ada Lovelace RTX 4070 Laptop GPU, 8GB VRAM).
Forward: Y = (X @ V_r.T) @ U_r.T + (X @ S.T if S is present) + bias
Backward computes exact analytical adjoint gradients for X, V_r, U_r, S, bias.
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
        with open(curr_dir / rpca_sparse_svd_binding.cpp, r, encoding=utf-8) as f:
            cpp_src = f.read()
        with open(curr_dir / rpca_sparse_svd.cu, r, encoding=utf-8) as f:
            cuda_src = f.read()

        _MODULE = load_cuda_module(
            name=cuda_rpca_sparse_svd,
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=[rpca_sparse_svd_fwd],
        )
    return _MODULE


def rpca_sparse_svd_gemm_pytorch_ref(
    x: torch.Tensor,
    v_r: torch.Tensor,
    u_r: torch.Tensor,
    s: Optional[Any] = None,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Pure PyTorch reference implementation."""
    from neuralops.core.triton_kernels.low_rank import triton_fused_lowrank_sparse_gemm_pytorch_ref
    return triton_fused_lowrank_sparse_gemm_pytorch_ref(x, v_r, u_r, s=s, bias=bias)


def _get_triton_impl():
    try:
        from neuralops.core.triton_kernels.low_rank import triton_fused_lowrank_sparse_gemm
        return triton_fused_lowrank_sparse_gemm
    except ImportError:
        return None


def _fwd_dispatch(x, v_r, u_r, s, bias):
    def cuda_fn(x, v_r, u_r, s=None, bias=None, **kwargs):
        mod = _get_module()
        s_tensor = s if isinstance(s, torch.Tensor) else None
        return mod.rpca_sparse_svd_fwd(
            x.contiguous(),
            v_r.contiguous(),
            u_r.contiguous(),
            s_tensor,
            bias.contiguous() if bias is not None else None,
        )

    R = v_r.shape[1] if v_r.shape[0] == x.shape[-1] else v_r.shape[0]
    if R <= 128 and x.is_cuda:
        try:
            return cuda_fn(x, v_r, u_r, s=s, bias=bias)
        except Exception:
            pass

    triton_fn = _get_triton_impl()
    if R > 128:
        cuda_fn = None

    return dispatch(
        cuda_fn,
        triton_fn,
        rpca_sparse_svd_gemm_pytorch_ref,
        x,
        v_r,
        u_r,
        s=s,
        bias=bias,
        kernel_name=rpca_sparse_svd_gemm,
    )


class _RPCASparseSVDGemmFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        v_r: torch.Tensor,
        u_r: torch.Tensor,
        s: torch.Tensor | None,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        orig_shape = x.shape
        K = orig_shape[-1]
        x_2d = x.reshape(-1, K)

        # Standardize matrix orientations:
        # V: [R, K], U: [N, R]
        V = v_r if v_r.shape[1] == K else v_r.t()
        R = V.shape[0]
        U = u_r if u_r.shape[1] == R else u_r.t()
        N = U.shape[0]

        out_2d = _fwd_dispatch(x_2d, V, U, s, bias)

        ctx.save_for_backward(x_2d, V, U, s if isinstance(s, torch.Tensor) else None)
        ctx.has_bias = bias is not None
        ctx.has_s = s is not None and isinstance(s, torch.Tensor)
        ctx.v_r_transposed = (v_r.shape[1] != K)
        ctx.u_r_transposed = (u_r.shape[1] != R)
        ctx.orig_shape = orig_shape
        ctx.out_features = N

        return out_2d.reshape(*orig_shape[:-1], N)

    @staticmethod
    def backward(ctx: Any, dout: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        x_2d, V, U, s = ctx.saved_tensors
        dout_2d = dout.reshape(-1, ctx.out_features)

        # Exact adjoint chain rule:
        # Y = (x @ V.T) @ U.T + (x @ S.T if S else 0) + bias
        # z = x @ V.T  [M, R]
        # dout_2d: [M, N]
        z = torch.matmul(x_2d, V.t())

        # dz = dout_2d @ U  [M, R]
        dz = torch.matmul(dout_2d, U)

        # dx = dz @ V + (dout_2d @ S if S else 0)  [M, K]
        dx_2d = torch.matmul(dz, V)
        ds = None
        if ctx.has_s:
            if s.is_sparse:
                s_dense = s.to_dense()
                dx_2d = dx_2d + torch.matmul(dout_2d, s_dense)
                if s.requires_grad:
                    ds = torch.matmul(dout_2d.t(), x_2d).to_sparse()
            else:
                dx_2d = dx_2d + torch.matmul(dout_2d, s)
                if s.requires_grad:
                    ds = torch.matmul(dout_2d.t(), x_2d)

        dx = dx_2d.reshape(ctx.orig_shape)

        dV = None
        if ctx.needs_input_grad[1]:
            dV = torch.matmul(dz.t(), x_2d)
            if ctx.v_r_transposed:
                dV = dV.t()

        dU = None
        if ctx.needs_input_grad[2]:
            dU = torch.matmul(dout_2d.t(), z)
            if ctx.u_r_transposed:
                dU = dU.t()

        dbias = None
        if ctx.has_bias and ctx.needs_input_grad[4]:
            dbias = dout_2d.sum(dim=0)

        return dx, dV, dU, ds, dbias


def rpca_sparse_svd_gemm(
    x: torch.Tensor,
    v_r: torch.Tensor,
    u_r: torch.Tensor,
    s: Optional[Any] = None,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """3-Level Dispatched RPCA Low-Rank + Sparse GEMM with full autograd support."""
    needs_grad = torch.is_grad_enabled() and any(
        t.requires_grad for t in (x, v_r, u_r, s, bias) if isinstance(t, torch.Tensor)
    )

    if needs_grad:
        return _RPCASparseSVDGemmFunction.apply(x, v_r, u_r, s, bias)

    return _fwd_dispatch(x, v_r, u_r, s, bias)
