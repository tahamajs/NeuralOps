"""
RMS Variance Match Autograd Interface

Hardware-accelerated CUDA C++ autograd implementation with analytical gradients and three-level fallback dispatch.

Author: Taha Majlesi
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F

from neuralops.core.cuda_kernels._bindings import load_cuda_module
from neuralops.core.cuda_kernels._fallback import dispatch

_MODULE = None


def _get_module():
    global _MODULE
    if _MODULE is None:
        curr_dir = Path(__file__).parent
        with open(curr_dir / "rms_variance_match_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "rms_variance_match.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()
        _MODULE = load_cuda_module(
            name="cuda_rms_variance_match",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["rms_variance_match_fwd"],
        )
    return _MODULE


from neuralops.core.triton_kernels.elementwise import fused_rms_variance_match_triton


def rms_variance_match_pytorch_ref(
    y: torch.Tensor,
    v_ref: torch.Tensor,
    clamp_min: float = 0.25,
    clamp_max: float = 4.0,
) -> torch.Tensor:
    """Pure PyTorch reference implementation."""
    v_rms = torch.sqrt(torch.mean(v_ref.float() ** 2, dim=-1, keepdim=True).clamp_min(1e-8))
    y_rms = torch.sqrt(torch.mean(y.float() ** 2, dim=-1, keepdim=True).clamp_min(1e-8))
    gain = (v_rms / y_rms).clamp(clamp_min, clamp_max).to(y.dtype)
    return y * gain


def _fwd_dispatch(y, v_ref, clamp_min=0.25, clamp_max=4.0):
    def cuda_fn(y, v_ref, clamp_min=0.25, clamp_max=4.0):
        mod = _get_module()
        return mod.rms_variance_match_fwd(
            y.contiguous(), v_ref.contiguous(), float(clamp_min), float(clamp_max)
        )

    return dispatch(
        cuda_fn,
        fused_rms_variance_match_triton,
        rms_variance_match_pytorch_ref,
        y,
        v_ref,
        clamp_min=clamp_min,
        clamp_max=clamp_max,
        kernel_name=rms_variance_match,
    )


class _RMSVarianceMatchFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        y: torch.Tensor,
        v_ref: torch.Tensor,
        clamp_min: float,
        clamp_max: float,
    ) -> torch.Tensor:
        out = _fwd_dispatch(y, v_ref, clamp_min=clamp_min, clamp_max=clamp_max)
        ctx.save_for_backward(y, v_ref)
        ctx.clamp_min = clamp_min
        ctx.clamp_max = clamp_max
        return out

    @staticmethod
    def backward(ctx: Any, dout: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        y, v_ref = ctx.saved_tensors
        clamp_min = ctx.clamp_min
        clamp_max = ctx.clamp_max

        # Analytical adjoint derivation:
        # v_sq = mean(v_ref^2, -1), v_rms = sqrt(v_sq)
        # y_sq = mean(y^2, -1), y_rms = sqrt(y_sq)
        # ratio = v_rms / y_rms
        # active = (ratio > clamp_min) & (ratio < clamp_max)
        # gain = clamp(ratio, clamp_min, clamp_max)
        # out = y * gain
        # dL/dy:
        # out_i = y_i * gain
        # dout_i * gain + y_i * (d gain / dy_i)
        # When active:
        # d gain / dy_i = - (v_rms / y_rms^2) * (d y_rms / dy_i)
        # d y_rms / dy_i = (1 / (2 * y_rms)) * (2 * y_i / D) = y_i / (D * y_rms)
        # => d gain / dy_i = - (v_rms / (D * y_rms^3)) * y_i
        # sum_i [dout_i * y_i * (- v_rms / (D * y_rms^3)) * y_k] ...
        # d gain / dv_ref_j = (1 / y_rms) * (v_ref_j / (D * v_rms)) = v_ref_j / (D * y_rms * v_rms)

        y_f = y.float()
        v_f = v_ref.float()
        dout_f = dout.float()
        D = y_f.shape[-1]

        v_sq = torch.mean(v_f ** 2, dim=-1, keepdim=True).clamp_min(1e-8)
        y_sq = torch.mean(y_f ** 2, dim=-1, keepdim=True).clamp_min(1e-8)
        v_rms = torch.sqrt(v_sq)
        y_rms = torch.sqrt(y_sq)

        ratio = v_rms / y_rms
        active = ((ratio > clamp_min) & (ratio < clamp_max)).float()
        gain = ratio.clamp(clamp_min, clamp_max)

        dy = None
        if ctx.needs_input_grad[0]:
            # term 1: dout * gain
            term1 = dout_f * gain
            # term 2: sum(dout * y, -1, keepdim) * (- v_rms / (D * y_rms^3)) * y * active
            dot_y = (dout_f * y_f).sum(dim=-1, keepdim=True)
            term2 = active * (- v_rms / (D * (y_rms ** 3))) * dot_y * y_f
            dy = (term1 + term2).to(y.dtype)

        dv_ref = None
        if ctx.needs_input_grad[1]:
            # dL/dv_ref = dot_y * (active / (D * y_rms * v_rms)) * v_ref
            dot_y = (dout_f * y_f).sum(dim=-1, keepdim=True)
            dv_ref = (active * (dot_y / (D * y_rms * v_rms)) * v_f).to(v_ref.dtype)

        return dy, dv_ref, None, None


def rms_variance_match(
    y: torch.Tensor,
    v_ref: torch.Tensor,
    clamp_min: float = 0.25,
    clamp_max: float = 4.0,
) -> torch.Tensor:
    """3-Level Dispatched RMS Variance Match with full autograd support."""
    needs_grad = torch.is_grad_enabled() and any(
        t.requires_grad for t in (y, v_ref) if isinstance(t, torch.Tensor)
    )

    if needs_grad:
        return _RMSVarianceMatchFunction.apply(y, v_ref, clamp_min, clamp_max)

    return _fwd_dispatch(y, v_ref, clamp_min=clamp_min, clamp_max=clamp_max)
