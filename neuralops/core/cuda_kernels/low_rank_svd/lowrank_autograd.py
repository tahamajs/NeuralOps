"""
Low-Rank SVD Gated GEMM Python Interface, Autograd Functions, and 3-Level Fallback.
Target: SM 8.9 (NVIDIA Ada Lovelace RTX 4070 Laptop GPU, 8GB VRAM).
Computes Y = (X @ V_r.T * sigmoid(alpha)) @ U_r.T + bias in a single register/shared-memory pass.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

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
) -> torch.Tensor:
    """Pure PyTorch reference implementation."""
    z = torch.matmul(x, V_r.t())
    if alpha is not None:
        z = z * torch.sigmoid(alpha)
    y = torch.matmul(z, U_r.t())
    if b_drift is not None:
        y = y + b_drift
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
        mod = _get_module()
        dout = dout.contiguous()
        dX = mod.fused_lowrank_gated_gemm_bwd(dout, x, V, U, a_arg)

        # PyTorch analytical fallback for dV, dU, dalpha, dbias if grads needed
        dV = None
        dU = None
        dalpha = None
        dbias = None

        if V.requires_grad or U.requires_grad or (ctx.has_alpha and a_arg.requires_grad):
            z = torch.matmul(x, V.t())
            if ctx.has_alpha:
                sig = torch.sigmoid(a_arg)
                z_gated = z * sig
            else:
                z_gated = z
            
            if U.requires_grad:
                dU = torch.matmul(dout.t(), z_gated)
            
            dz = torch.matmul(dout, U)
            if ctx.has_alpha:
                if a_arg.requires_grad:
                    dalpha = (dz * z * sig * (1.0 - sig)).sum(dim=0)
                dz = dz * sig
            
            if V.requires_grad:
                dV = torch.matmul(dz.t(), x)

        if ctx.has_bias:
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
    def cuda_fn(x, V, U, alpha=None, bias=None):
        return _FusedLowRankGatedGemmFunction.apply(x, V, U, alpha, bias)

    triton_fn = _get_triton_impl()

    return dispatch(
        cuda_fn=cuda_fn,
        triton_fn=triton_fn,
        pytorch_fn=fused_lowrank_gated_gemm_pytorch_ref,
        args=(x, V, U),
        kwargs={"alpha": alpha, "bias": bias if bias is not None else None},
        op_name="fused_lowrank_gated_gemm",
    )
