"""
Quantized GEMM Autograd & 3-Level Fallback (Module 3).
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
        with open(curr_dir / "quant_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "w4a16_gemm.cu", "r", encoding="utf-8") as f:
            w4_src = f.read()
        with open(curr_dir / "fp8_gemm.cu", "r", encoding="utf-8") as f:
            fp8_src = f.read()

        cuda_combined = w4_src + "\n" + fp8_src

        _MODULE = load_cuda_module(
            name="cuda_quant_gemm",
            cpp_sources=cpp_src,
            cuda_sources=cuda_combined,
            functions=["w4a16_gemm", "fp8_gemm"],
        )
    return _MODULE


def w4a16_gemm_pytorch_ref(
    A: torch.Tensor,
    W_q: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    M, K = A.shape
    num_groups, N = scales.shape

    # Unpack int4 [K, N/2] -> [K, N]
    low = (W_q & 0x0F).to(torch.float32)
    high = ((W_q >> 4) & 0x0F).to(torch.float32)
    W_unpacked = torch.empty(K, N, device=A.device, dtype=torch.float32)
    W_unpacked[:, 0::2] = low
    W_unpacked[:, 1::2] = high

    scales_expanded = scales.repeat_interleave(group_size, dim=0)[:K, :]
    zeros_expanded = zeros.repeat_interleave(group_size, dim=0)[:K, :]

    W_dequant = (W_unpacked - zeros_expanded) * scales_expanded
    return torch.matmul(A, W_dequant)


def w4a16_gemm(
    A: torch.Tensor,
    W_q: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    mod = _get_module()
    cuda_fn = mod.w4a16_gemm if mod is not None else None
    return dispatch(
        cuda_fn,
        None,
        w4a16_gemm_pytorch_ref,
        A,
        W_q,
        scales,
        zeros,
        group_size,
        kernel_name="w4a16_gemm",
    )


def fp8_gemm_pytorch_ref(
    A: torch.Tensor,
    B: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
) -> torch.Tensor:
    return torch.matmul(A.float(), B.float()) * (scale_a.item() * scale_b.item())


def fp8_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
) -> torch.Tensor:
    mod = _get_module()
    cuda_fn = mod.fp8_gemm if mod is not None else None
    return dispatch(
        cuda_fn,
        None,
        fp8_gemm_pytorch_ref,
        A,
        B,
        scale_a,
        scale_b,
        kernel_name="fp8_gemm",
    )
