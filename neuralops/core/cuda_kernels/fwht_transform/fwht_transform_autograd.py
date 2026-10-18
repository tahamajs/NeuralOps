"""
Fast Walsh-Hadamard Transform Autograd Interface

Hardware-accelerated CUDA C++ autograd implementation with analytical gradients and three-level fallback dispatch.

Author: Taha Majlesi
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional
import torch
import torch.nn.functional as F

from neuralops.core.cuda_kernels._bindings import load_cuda_module
from neuralops.core.cuda_kernels._fallback import dispatch

_MODULE = None

def _get_module():
    global _MODULE
    if _MODULE is None:
        curr_dir = Path(__file__).parent
        with open(curr_dir / "fwht_transform_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "fwht_transform.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()
        _MODULE = load_cuda_module(
            name="cuda_fwht_transform",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["fwht_fwd"],
        )
    return _MODULE

from neuralops.core.triton_kernels.elementwise import fast_hadamard_transform

def fwht_transform_pytorch_ref(x: torch.Tensor, B_dim: int = 128) -> torch.Tensor:
    return fast_hadamard_transform(x, block_size=B_dim)

def fwht_transform(x: torch.Tensor, B_dim: int = 128) -> torch.Tensor:
    if (B_dim & (B_dim - 1)) != 0:
        raise ValueError(f"fwht_transform: B_dim={B_dim} must be a power of 2")
    def cuda_fn(x, B_dim=128):
        mod = _get_module()
        return mod.fwht_fwd(x.contiguous(), B_dim)
    return dispatch(
        cuda_fn,
        fwht_transform_pytorch_ref,
        fwht_transform_pytorch_ref,
        x, B_dim=B_dim,
        kernel_name="fwht_transform",
    )
