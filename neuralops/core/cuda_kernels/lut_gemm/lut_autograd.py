"""
LUT Codebook Dequantization GEMM Autograd Interface

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
        with open(curr_dir / "lut_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "lut_gemm.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()

        _MODULE = load_cuda_module(
            name="cuda_lut_gemm",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["lut_codebook_dequant_gemm_fwd"],
        )
    return _MODULE


def lut_codebook_dequant_gemm_pytorch_ref(
    x: torch.Tensor,
    indices_packed: torch.Tensor,
    codebook: torch.Tensor,
    group_size: int = 64,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pure PyTorch reference implementation."""
    from neuralops.core.triton_kernels.fused_norm_linear import triton_lut_codebook_dequant_gemm_pytorch_ref
    return triton_lut_codebook_dequant_gemm_pytorch_ref(
        x, indices_packed, codebook, group_size=group_size, bias=bias
    )


def _get_triton_impl():
    try:
        from neuralops.core.triton_kernels.fused_norm_linear import triton_lut_codebook_dequant_gemm
        return triton_lut_codebook_dequant_gemm
    except ImportError:
        return None


lut_gemm_ref = lut_codebook_dequant_gemm_pytorch_ref
lut_gemm_pytorch_ref = lut_codebook_dequant_gemm_pytorch_ref


def lut_codebook_dequant_gemm(
    x: torch.Tensor,
    indices_packed: torch.Tensor,
    codebook: torch.Tensor,
    group_size: int = 64,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """3-Level Dispatched LUT Codebook Dequantization GEMM."""
    def cuda_fn(x, indices_packed, codebook, group_size=64, bias=None):
        mod = _get_module()
        dummy_bias = torch.empty(0, device=x.device, dtype=x.dtype)
        b_arg = bias.contiguous() if bias is not None else dummy_bias
        cb = codebook
        if cb.dim() == 1:
            K = indices_packed.shape[1] * 2
            num_groups = max(1, K // group_size)
            cb = cb.unsqueeze(0).expand(num_groups, 16).contiguous()
        return mod.lut_codebook_dequant_gemm_fwd(
            x.contiguous(), indices_packed.contiguous(), cb.contiguous(), group_size, b_arg
        )

    triton_fn = _get_triton_impl()

    return dispatch(
        cuda_fn,
        triton_fn,
        lut_codebook_dequant_gemm_pytorch_ref,
        x,
        indices_packed,
        codebook,
        group_size=group_size,
        bias=bias,
        kernel_name="lut_codebook_dequant_gemm",
    )
