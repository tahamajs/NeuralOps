"""
Fused Softplus Interface and 3-Level Fallback.
Target: SM 8.9 (NVIDIA Ada Lovelace RTX 4070 Laptop GPU, 8GB VRAM).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from neuralops.core.cuda_kernels._bindings import load_cuda_module
from neuralops.core.cuda_kernels._fallback import dispatch

_MODULE = None


def _get_module():
    global _MODULE
    if _MODULE is None:
        curr_dir = Path(__file__).parent
        with open(curr_dir / "softplus_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "softplus.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()

        _MODULE = load_cuda_module(
            name="cuda_softplus",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["softplus_fwd"],
        )
    return _MODULE


def fused_softplus_pytorch_ref(
    x: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> torch.Tensor:
    """Pure PyTorch reference implementation."""
    return F.softplus(x, beta=beta, threshold=threshold)


def _get_triton_impl():
    try:
        from neuralops.core.triton_kernels.softplus import softplus_triton
        return softplus_triton
    except (ImportError, AttributeError):
        return None


def fused_softplus(
    x: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> torch.Tensor:
    """3-Level Dispatched Fused Softplus."""
    def cuda_fn(x, beta=1.0, threshold=20.0, **kwargs):
        mod = _get_module()
        return mod.softplus_fwd(x.contiguous(), float(beta), float(threshold))

    triton_fn = _get_triton_impl()

    return dispatch(
        cuda_fn,
        triton_fn,
        fused_softplus_pytorch_ref,
        x,
        beta=beta,
        threshold=threshold,
        kernel_name="fused_softplus",
    )
