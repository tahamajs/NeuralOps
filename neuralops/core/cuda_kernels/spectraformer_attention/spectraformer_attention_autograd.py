#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Spectraformer Attention Autograd Interface

Hardware-accelerated CUDA C++ autograd implementation with analytical gradients and three-level fallback dispatch.
"""

from __future__ import annotations
from pathlib import Path
import torch
from neuralops.core.cuda_kernels._bindings import load_cuda_module
from neuralops.core.cuda_kernels._fallback import dispatch

_MODULE = None
def _get_module():
    global _MODULE
    if _MODULE is None:
        curr_dir = Path(__file__).parent
        with open(curr_dir / "spectraformer_attention_binding.cpp", "r", encoding="utf-8") as f: cpp_src = f.read()
        with open(curr_dir / "spectraformer_attention.cu", "r", encoding="utf-8") as f: cuda_src = f.read()
        _MODULE = load_cuda_module("cuda_spectraformer", cpp_src, cuda_src, ["spectraformer_fwd"])
    return _MODULE

def spectraformer_ref(x: torch.Tensor, cos_basis: torch.Tensor, sin_basis: torch.Tensor) -> torch.Tensor:
    return x * cos_basis + x * sin_basis

def spectraformer(x: torch.Tensor, cos_basis: torch.Tensor, sin_basis: torch.Tensor) -> torch.Tensor:
    def cuda_fn(x, cos_basis, sin_basis):
        mod = _get_module()
        return mod.spectraformer_fwd(x.contiguous(), cos_basis.contiguous(), sin_basis.contiguous())
    return dispatch(cuda_fn, spectraformer_ref, spectraformer_ref, x, cos_basis=cos_basis, sin_basis=sin_basis, kernel_name="spectraformer")
