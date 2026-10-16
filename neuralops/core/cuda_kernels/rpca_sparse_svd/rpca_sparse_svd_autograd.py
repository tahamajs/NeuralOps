"""
RPCA Low-Rank + Sparse SVD GEMM Interface and 3-Level Fallback.
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
        with open(curr_dir / "rpca_sparse_svd_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "rpca_sparse_svd.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()

        _MODULE = load_cuda_module(
            name="cuda_rpca_sparse_svd",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["rpca_sparse_svd_fwd"],
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


def rpca_sparse_svd_gemm(
    x: torch.Tensor,
    v_r: torch.Tensor,
    u_r: torch.Tensor,
    s: Optional[Any] = None,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """3-Level Dispatched RPCA Low-Rank + Sparse GEMM."""
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

    triton_fn = _get_triton_impl()

    return dispatch(
        cuda_fn,
        triton_fn,
        rpca_sparse_svd_gemm_pytorch_ref,
        x,
        v_r,
        u_r,
        s=s,
        bias=bias,
        kernel_name="rpca_sparse_svd_gemm",
    )
