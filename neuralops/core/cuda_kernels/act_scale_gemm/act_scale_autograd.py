"""
Fused Activation Scaling GEMM Interface and 3-Level Fallback.
Target: SM 8.9 (NVIDIA Ada Lovelace RTX 4070 Laptop GPU, 8GB VRAM).
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
        with open(curr_dir / "act_scale_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "act_scale_gemm.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()

        _MODULE = load_cuda_module(
            name="cuda_act_scale_gemm",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["act_scale_quant_gemm_fwd"],
        )
    return _MODULE


def act_scale_quant_gemm_pytorch_ref(
    x: torch.Tensor,
    s_inv: torch.Tensor,
    w: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Pure PyTorch reference implementation."""
    from neuralops.core.triton_kernels.fused_norm_linear import triton_fused_act_scale_quant_gemm_pytorch_ref
    return triton_fused_act_scale_quant_gemm_pytorch_ref(x, s_inv, w, bias=bias)


def _get_triton_impl():
    try:
        from neuralops.core.triton_kernels.fused_norm_linear import triton_fused_act_scale_quant_gemm
        return triton_fused_act_scale_quant_gemm
    except ImportError:
        return None


def act_scale_quant_gemm(
    x: torch.Tensor,
    s_inv: torch.Tensor,
    w: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """3-Level Dispatched Act-Scale GEMM."""
    def cuda_fn(x, s_inv, w, bias=None, **kwargs):
        mod = _get_module()
        return mod.act_scale_quant_gemm_fwd(
            x.contiguous(),
            s_inv.contiguous().to(x.dtype),
            w.contiguous(),
            bias.contiguous() if bias is not None else None,
        )

    triton_fn = _get_triton_impl()

    return dispatch(
        cuda_fn,
        triton_fn,
        act_scale_quant_gemm_pytorch_ref,
        x,
        s_inv,
        w,
        bias=bias,
        kernel_name="act_scale_quant_gemm",
    )
