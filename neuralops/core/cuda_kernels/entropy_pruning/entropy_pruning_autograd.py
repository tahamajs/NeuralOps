#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Entropy Pruning Autograd Interface

Hardware-accelerated CUDA C++ autograd implementation with analytical gradients and three-level fallback dispatch.
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
        with open(curr_dir / "entropy_pruning_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "entropy_pruning.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()
        _MODULE = load_cuda_module(
            name="cuda_entropy_pruning",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["entropy_row_reduction_fwd"],
        )
    return _MODULE

def entropy_pruning_pytorch_ref(scores):
    p = torch.softmax(scores.float(), dim=-1)
    return (-p * torch.log(p.clamp(min=1e-8))).sum(dim=-1).to(scores.dtype)
def entropy_pruning_triton_ref(scores):
    from neuralops.core.triton_kernels.elementwise import fused_entropy_token_pruning_mask_triton
    _, h = fused_entropy_token_pruning_mask_triton(scores)
    return h

def entropy_pruning(scores: torch.Tensor) -> torch.Tensor:
    orig_shape = scores.shape[:-1]
    def cuda_fn(*args, **kwargs):
        mod = _get_module()
        fn = mod.entropy_row_reduction_fwd
        res = fn(*args, **kwargs)
        return res.view(*orig_shape) if len(orig_shape) > 1 else res
    return dispatch(
        cuda_fn,
        entropy_pruning_triton_ref,
        entropy_pruning_pytorch_ref,
        scores,
        kernel_name="entropy_pruning",
    )
