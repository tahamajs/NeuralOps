"""
Bitonic Top-K Selection Autograd Interface

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
        with open(curr_dir / "bitonic_topk_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "bitonic_topk.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()
        _MODULE = load_cuda_module(
            name="cuda_bitonic_topk",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["bitonic_topk_indices_fwd"],
        )
    return _MODULE


def bitonic_topk_pytorch_ref(x: torch.Tensor, K: int) -> torch.Tensor:
    """Pure PyTorch reference implementation."""
    return torch.topk(x, k=K, dim=-1).indices


def bitonic_topk_triton_ref(x: torch.Tensor, K: int) -> torch.Tensor:
    return bitonic_topk_pytorch_ref(x, K)


def _fwd_dispatch(x: torch.Tensor, K: int) -> torch.Tensor:
    def cuda_fn(*args, **kwargs):
        mod = _get_module()
        fn = getattr(mod, "bitonic_topk_indices_fwd")
        return fn(*args, **kwargs)

    return dispatch(
        cuda_fn,
        bitonic_topk_triton_ref,
        bitonic_topk_pytorch_ref,
        x,
        K,
        kernel_name="bitonic_topk",
    )


class _BitonicTopKFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, K: int) -> torch.Tensor:
        return _fwd_dispatch(x, K)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> tuple[torch.Tensor | None, ...]:
        # Indices are integer/discrete tensors with zero continuous gradient
        return None, None


def bitonic_topk(x: torch.Tensor, K: int) -> torch.Tensor:
    """3-Level Dispatched Bitonic Top-K with autograd function support."""
    if x.requires_grad:
        return _BitonicTopKFunction.apply(x, K)
    return _fwd_dispatch(x, K)
