#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Fused SVD Projection CUDA Autograd & 3-Level Fallback Dispatcher.

Implements:
    Y = alpha * ((X * U) * diag(S) * V_T) + beta * X_res + bias
with full analytical autograd backward support across (X, U, S, V_T, X_res, bias).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch.autograd import Function

from neuralops.ops.cuda import _fallback
from neuralops.ops.cuda._bindings import load_cuda_module

logger = logging.getLogger(__name__)

_MODULE = None


def _get_cuda_module():
    global _MODULE
    if _MODULE is None:
        curr_dir = Path(__file__).parent
        with open(curr_dir / "fused_svd_projection_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "fused_svd_projection.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()

        _MODULE = load_cuda_module(
            name="cuda_fused_svd_projection",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["fused_svd_projection_fwd", "fused_svd_projection_bwd_dx"],
        )
    return _MODULE


def fused_svd_projection_pytorch_ref(
    x: torch.Tensor,
    U: torch.Tensor,
    S: torch.Tensor | None = None,
    V_T: torch.Tensor | None = None,
    X_res: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> torch.Tensor:
    """Vectorized PyTorch Reference (Level 3 Fallback)."""
    orig_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1])
    orig_dtype = x.dtype

    # Down-projection: Z = X @ U
    z = torch.matmul(x_2d.float(), U.float())
    if S is not None:
        z = z * S.float()

    # Up-projection: Out = alpha * (Z @ V_T)
    out = alpha * torch.matmul(z, V_T.float())

    if X_res is not None:
        x_res_2d = X_res.reshape(-1, X_res.shape[-1])
        out = out + beta * x_res_2d.float()

    if bias is not None:
        out = out + bias.float()

    out = out.to(orig_dtype)
    return out.reshape(*orig_shape[:-1], out.shape[-1])


class FusedSVDProjectionFunction(Function):
    """Autograd Function for Fused SVD Projection."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        U: torch.Tensor,
        S: torch.Tensor | None,
        V_T: torch.Tensor,
        X_res: torch.Tensor | None,
        bias: torch.Tensor | None,
        alpha: float,
        beta: float,
    ) -> torch.Tensor:
        orig_shape = x.shape
        x_2d = x.reshape(-1, x.size(-1)).contiguous()
        U_c = U.contiguous()
        V_T_c = V_T.contiguous()
        S_c = S.contiguous() if (S is not None and S.numel() > 0) else None
        res_2d = X_res.reshape(-1, X_res.size(-1)).contiguous() if (X_res is not None and X_res.numel() > 0) else None
        bias_c = bias.contiguous() if (bias is not None and bias.numel() > 0) else None

        ctx.has_s = S_c is not None
        ctx.has_res = res_2d is not None
        ctx.has_bias = bias_c is not None
        ctx.alpha = float(alpha)
        ctx.beta = float(beta)
        ctx.orig_shape = orig_shape

        dispatched = False
        out = None

        if x.is_cuda:
            try:
                mod = _get_cuda_module()
                if mod is not None:
                    dummy_s = S_c if ctx.has_s else torch.empty(0, device=x.device, dtype=x.dtype)
                    dummy_res = res_2d if ctx.has_res else torch.empty(0, device=x.device, dtype=x.dtype)
                    dummy_b = bias_c if ctx.has_bias else torch.empty(0, device=x.device, dtype=x.dtype)

                    out = mod.fused_svd_projection_fwd(
                        x_2d, U_c, dummy_s, V_T_c, dummy_res, dummy_b, ctx.alpha, ctx.beta
                    )
                    dispatched = True
                    _fallback.set_last_dispatch_backend("fused_svd_projection", "cuda")
            except Exception as e:
                logger.debug("FusedSVDProjection CUDA forward failed (%s); falling back to Level 2/3.", e)

        # Level 2: Triton JIT Kernel
        if not dispatched and x.is_cuda:
            try:
                from neuralops.ops.triton.fused_svd_projection_triton import fused_svd_projection_triton

                out = fused_svd_projection_triton(
                    x_2d, U_c, S_c, V_T_c, res_2d, bias_c, alpha=ctx.alpha, beta=ctx.beta
                )
                dispatched = True
                _fallback.set_last_dispatch_backend("fused_svd_projection", "triton")
            except Exception as e:
                logger.debug("FusedSVDProjection Triton forward failed (%s); falling back to Level 3.", e)

        # Level 3: Vectorized PyTorch Reference
        if not dispatched:
            out = fused_svd_projection_pytorch_ref(
                x_2d, U_c, S_c, V_T_c, res_2d, bias_c, alpha=ctx.alpha, beta=ctx.beta
            )
            _fallback.set_last_dispatch_backend("fused_svd_projection", "pytorch")

        ctx.save_for_backward(x_2d, U_c, S_c, V_T_c, res_2d, bias_c)
        return out.reshape(*orig_shape[:-1], out.shape[-1])

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x_2d, U, S, V_T, res_2d, bias = ctx.saved_tensors
        dout = grad_output.reshape(-1, grad_output.size(-1)).contiguous()

        needs_dx = ctx.needs_input_grad[0]
        needs_dU = ctx.needs_input_grad[1]
        needs_dS = ctx.needs_input_grad[2]
        needs_dVT = ctx.needs_input_grad[3]
        needs_dres = ctx.needs_input_grad[4]
        needs_dbias = ctx.needs_input_grad[5]

        dx = None
        dU = None
        dS = None
        dV_T = None
        dres = None
        dbias = None

        # Try CUDA kernel for dX if on CUDA
        dx_dispatched = False
        if needs_dx and dout.is_cuda:
            try:
                mod = _get_cuda_module()
                if mod is not None:
                    dummy_s = (
                        S if (S is not None and S.numel() > 0) else torch.empty(0, device=dout.device, dtype=dout.dtype)
                    )
                    dx = mod.fused_svd_projection_bwd_dx(dout, U, dummy_s, V_T, ctx.alpha)
                    dx = dx.reshape(ctx.orig_shape)
                    dx_dispatched = True
            except Exception:
                dx_dispatched = False

        # Analytic autograd gradient computation
        dout_f = dout.float()
        x_f = x_2d.float()
        U_f = U.float()
        V_T_f = V_T.float()

        # Intermediate: Z = X * U, Z_scaled = Z * S
        z = torch.matmul(x_f, U_f)
        z_scaled = z * S.float() if ctx.has_s else z

        # Backward for dV_T: dOut^T * Z_scaled
        if needs_dVT:
            dV_T = (ctx.alpha * torch.matmul(z_scaled.t(), dout_f)).to(V_T.dtype)

        # dz_scaled = alpha * (dOut * V_T^T)
        dz_scaled = ctx.alpha * torch.matmul(dout_f, V_T_f.t())

        # Backward for dS: sum(dz_scaled * Z) along M
        if needs_dS and ctx.has_s:
            dS = (dz_scaled * z).sum(dim=0).to(S.dtype)

        # dz = dz_scaled * S
        dz = dz_scaled * S.float() if ctx.has_s else dz_scaled

        # Backward for dU: X^T * dz
        if needs_dU:
            dU = torch.matmul(x_f.t(), dz).to(U.dtype)

        # Backward for dX if not already computed via CUDA
        if needs_dx and not dx_dispatched:
            dx_2d = torch.matmul(dz, U_f.t()).to(x_2d.dtype)
            dx = dx_2d.reshape(ctx.orig_shape)

        # Backward for X_res
        if needs_dres and ctx.has_res:
            dres_2d = (ctx.beta * dout).to(res_2d.dtype)
            dres = dres_2d.reshape(ctx.orig_shape)

        # Backward for bias
        if needs_dbias and ctx.has_bias:
            dbias = dout.sum(dim=0).to(bias.dtype)

        return dx, dU, dS, dV_T, dres, dbias, None, None


def fused_svd_projection(
    x: torch.Tensor,
    U: torch.Tensor,
    S: torch.Tensor | None = None,
    V_T: torch.Tensor | None = None,
    X_res: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> torch.Tensor:
    """
    3-Level Dispatched Fused SVD Projection Interface:
        Level 1: Native CUDA C++ (fused_svd_projection_fwd)
        Level 2: Triton JIT (fused_svd_projection_triton)
        Level 3: Vectorized PyTorch Reference (fused_svd_projection_pytorch_ref)
    """
    forced = getattr(_fallback, "_FORCED_BACKEND", None)
    backend_val = forced.get() if forced is not None else "auto"

    if backend_val == "pytorch" or not x.is_cuda:
        return fused_svd_projection_pytorch_ref(x, U, S=S, V_T=V_T, X_res=X_res, bias=bias, alpha=alpha, beta=beta)

    if backend_val == "triton":
        from neuralops.ops.triton.fused_svd_projection_triton import fused_svd_projection_triton

        return fused_svd_projection_triton(x, U, S=S, V_T=V_T, X_res=X_res, bias=bias, alpha=alpha, beta=beta)

    # Use Autograd Function for full gradient propagation
    return FusedSVDProjectionFunction.apply(x, U, S, V_T, X_res, bias, alpha, beta)


__all__ = [
    "fused_svd_projection",
    "FusedSVDProjectionFunction",
    "fused_svd_projection_pytorch_ref",
]
