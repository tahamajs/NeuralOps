"""
Hadamard W4A16 GEMM Interface and 3-Level Fallback.
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
        with open(curr_dir / "hadamard_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "hadamard_gemm.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()

        _MODULE = load_cuda_module(
            name="cuda_hadamard_gemm",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["hadamard_w4a16_gemm_fwd"],
        )
    return _MODULE


def hadamard_w4a16_gemm_pytorch_ref(
    x: torch.Tensor,
    w_packed: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    group_size: int = 32,
    bias: Optional[torch.Tensor] = None,
    apply_hadamard: bool = False,
) -> torch.Tensor:
    """Pure PyTorch reference implementation."""
    from neuralops.core.triton_kernels.fused_norm_linear import triton_fused_hadamard_w4a16_gemm_pytorch_ref
    return triton_fused_hadamard_w4a16_gemm_pytorch_ref(
        x, w_packed, scales, zeros, group_size=group_size, bias=bias, apply_hadamard=apply_hadamard
    )


def _get_triton_impl():
    try:
        from neuralops.core.triton_kernels.fused_norm_linear import triton_fused_hadamard_w4a16_gemm
        return triton_fused_hadamard_w4a16_gemm
    except ImportError:
        return None


def hadamard_w4a16_gemm(
    x: torch.Tensor,
    w_packed: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    group_size: int = 32,
    bias: Optional[torch.Tensor] = None,
    apply_hadamard: bool = False,
) -> torch.Tensor:
    """3-Level Dispatched Hadamard W4A16 GEMM."""
    def cuda_fn(x, w_packed, scales, zeros, group_size=32, bias=None, apply_hadamard=False, **kwargs):
        mod = _get_module()
        return mod.hadamard_w4a16_gemm_fwd(
            x.contiguous(),
            w_packed.contiguous(),
            scales.contiguous(),
            zeros.contiguous(),
            group_size,
            bias.contiguous() if bias is not None else None,
        )

    triton_fn = _get_triton_impl()

    return dispatch(
        cuda_fn,
        triton_fn,
        hadamard_w4a16_gemm_pytorch_ref,
        x,
        w_packed,
        scales,
        zeros,
        group_size=group_size,
        bias=bias,
        apply_hadamard=apply_hadamard,
        kernel_name="hadamard_w4a16_gemm",
    )
