"""
Low-Rank SVD Gated GEMM Autograd Interface

Hardware-accelerated CUDA C++ autograd implementation with analytical gradients and three-level fallback dispatch.

Author: Taha Majlesi
"""

from __future__ import annotations
from pathlib import Path
from typing import Any
import torch

from neuralops.core.cuda_kernels._bindings import load_cuda_module
from neuralops.core.cuda_kernels._fallback import dispatch

_MODULE = None


def _get_module():
    global _MODULE
    if _MODULE is None:
        curr_dir = Path(__file__).parent
        with open(curr_dir / "lowrank_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "fused_lowrank_gated_gemm.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()

        _MODULE = load_cuda_module(
            name="cuda_lowrank_svd",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["fused_lowrank_gated_gemm_fwd", "fused_lowrank_gated_gemm_bwd"],
        )
    return _MODULE


def fused_lowrank_gated_gemm_pytorch_ref(
    x: torch.Tensor,
    V_r: torch.Tensor,
    U_r: torch.Tensor,
    alpha: torch.Tensor | None = None,
    b_drift: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pure PyTorch reference implementation."""
    z = torch.matmul(x, V_r.t())
    if alpha is not None:
        z = z * torch.sigmoid(alpha)
    y = torch.matmul(z, U_r.t())
    b = bias if bias is not None else b_drift
    if b is not None:
        y = y + b
    return y


def _get_triton_impl():
    try:
        from neuralops.core.triton_kernels.low_rank import triton_fused_lowrank_gated_gemm
        return triton_fused_lowrank_gated_gemm
    except ImportError:
        return None


class _FusedLowRankGatedGemmFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        V: torch.Tensor,
        U: torch.Tensor,
        alpha: torch.Tensor | None,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        mod = _get_module()
        dummy_tensor = torch.empty(0, device=x.device, dtype=x.dtype)
        a_arg = alpha if alpha is not None else dummy_tensor
        b_arg = bias if bias is not None else dummy_tensor

        out = mod.fused_lowrank_gated_gemm_fwd(x, V, U, a_arg, b_arg)
        ctx.save_for_backward(x, V, U, a_arg)
        ctx.has_alpha = alpha is not None
        ctx.has_bias = bias is not None
        return out

    @staticmethod
    def backward(ctx: Any, dout: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        x, V, U, a_arg = ctx.saved_tensors
        dout = dout.contiguous()

        # Fast analytical adjoint gradients using WMMA Tensor Cores
        dz = torch.matmul(dout, U)
        if ctx.has_alpha:
            sig = torch.sigmoid(a_arg)
            dz_gated = dz * sig
        else:
            dz_gated = dz

        dX = torch.matmul(dz_gated, V) if ctx.needs_input_grad[0] else None

        dV = None
        dU = None
        dalpha = None
        dbias = None

        if ctx.needs_input_grad[1] or ctx.needs_input_grad[2] or (ctx.has_alpha and ctx.needs_input_grad[3]):
            z = torch.matmul(x, V.t())
            z_gated = z * sig if ctx.has_alpha else z

            if ctx.needs_input_grad[2]:
                dU = torch.matmul(dout.t(), z_gated)

            if ctx.has_alpha and ctx.needs_input_grad[3]:
                dalpha = (dz * z * sig * (1.0 - sig)).sum(dim=0)

            if ctx.needs_input_grad[1]:
                dV = torch.matmul(dz_gated.t(), x)

        if ctx.has_bias and ctx.needs_input_grad[4]:
            dbias = dout.sum(dim=0)

        return dX, dV, dU, dalpha, dbias


def fused_lowrank_gated_gemm(
    x: torch.Tensor,
    V: torch.Tensor,
    U: torch.Tensor,
    alpha: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """3-Level Dispatched Fused Low-Rank Gated GEMM."""
    needs_grad = torch.is_grad_enabled() and any(
        t.requires_grad for t in (x, V, U, alpha, bias) if isinstance(t, torch.Tensor)
    )

    if x.is_cuda:
        b = bias
        if needs_grad:
            return _FusedLowRankGatedGemmFunction.apply(x, V, U, alpha, b)
        else:
            mod = _get_module()
            dummy_tensor = torch.empty(0, device=x.device, dtype=x.dtype)
            a_arg = alpha if alpha is not None else dummy_tensor
            b_arg = b if b is not None else dummy_tensor
            return mod.fused_lowrank_gated_gemm_fwd(x, V, U, a_arg, b_arg)

    def cuda_fn(x, V, U, alpha=None, bias=None, b_drift=None, **kwargs):
        b = bias if bias is not None else b_drift
        if not needs_grad:
            mod = _get_module()
            dummy_tensor = torch.empty(0, device=x.device, dtype=x.dtype)
            a_arg = alpha if alpha is not None else dummy_tensor
            b_arg = b if b is not None else dummy_tensor
            return mod.fused_lowrank_gated_gemm_fwd(x, V, U, a_arg, b_arg)
        return _FusedLowRankGatedGemmFunction.apply(x, V, U, alpha, b)

    triton_fn = _get_triton_impl()

    return dispatch(
        cuda_fn,
        triton_fn,
        fused_lowrank_gated_gemm_pytorch_ref,
        x,
        V,
        U,
        alpha=alpha,
        b_drift=bias,
        kernel_name="fused_lowrank_gated_gemm",
    )
