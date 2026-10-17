"""
1.58-Bit Ternary GEMM Interface and 3-Level Fallback.
Target: SM 8.9 (NVIDIA Ada Lovelace RTX 4070 Laptop GPU, 8GB VRAM).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from neuralops.core.cuda_kernels._bindings import load_cuda_module
from neuralops.core.cuda_kernels._fallback import dispatch

_MODULE = None


def _get_module():
    global _MODULE
    if _MODULE is None:
        curr_dir = Path(__file__).parent
        with open(curr_dir / "ternary_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "ternary_gemm.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()

        _MODULE = load_cuda_module(
            name="cuda_ternary_gemm",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["ternary_gemm_fwd"],
        )
    return _MODULE


def ternary_gemm_pytorch_ref(
    x: torch.Tensor,
    w_packed: torch.Tensor,
    scales: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Pure PyTorch reference implementation."""
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    N, K_packed = w_packed.shape
    K = K_packed * 4

    w_unpacked = torch.empty((N, K), dtype=torch.float32, device=x.device)
    for i in range(4):
        code = (w_packed >> (2 * i)) & 0x03
        val = torch.where(code == 0, -1.0, torch.where(code == 2, 1.0, 0.0))
        w_unpacked[:, i::4] = val

    w_scaled = w_unpacked * scales.unsqueeze(-1)
    y = torch.matmul(x_2d, w_scaled.t().to(x.dtype))
    if bias is not None:
        y = y + bias
    return y.reshape(*orig_shape[:-1], N)


def _get_triton_impl():
    try:
        from neuralops.core.triton_kernels.fused_norm_linear import ternary_gemm_triton
        return ternary_gemm_triton
    except (ImportError, AttributeError):
        return None


def ternary_gemm(
    x: torch.Tensor,
    w_packed: torch.Tensor,
    scales: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """3-Level Dispatched Ternary GEMM."""
    def cuda_fn(x, w_packed, scales, bias=None, **kwargs):
        mod = _get_module()
        return mod.ternary_gemm_fwd(
            x.contiguous(),
            w_packed.contiguous(),
            scales.contiguous(),
            bias.contiguous() if bias is not None else None,
        )

    triton_fn = _get_triton_impl()

    return dispatch(
        cuda_fn,
        triton_fn,
        ternary_gemm_pytorch_ref,
        x,
        w_packed,
        scales,
        bias=bias,
        kernel_name="ternary_gemm",
    )
