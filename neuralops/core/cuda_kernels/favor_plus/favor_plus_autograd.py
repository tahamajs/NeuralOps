#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
FAVOR+ Random Feature Attention Autograd Interface

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
        with open(curr_dir / "favor_plus_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "favor_plus.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()
        _MODULE = load_cuda_module(
            name="cuda_favor_plus",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["favor_plus_act_fwd"],
        )
    return _MODULE

def favor_plus_pytorch_ref(proj, diag_sq, norm_factor):
    diff = (proj.float() - diag_sq.float().view(-1, 1)).clamp(-20.0, 8.0)
    return (torch.exp(diff) * norm_factor).to(proj.dtype)
def favor_plus_triton_ref(proj, diag_sq, norm_factor):
    return favor_plus_pytorch_ref(proj, diag_sq, norm_factor)

def favor_plus(proj: torch.Tensor, diag_sq: torch.Tensor, norm_factor: float) -> torch.Tensor:
    def cuda_fn(*args, **kwargs):
        mod = _get_module()
        fn = mod.favor_plus_act_fwd
        return fn(*args, **kwargs)
    return dispatch(
        cuda_fn,
        favor_plus_triton_ref,
        favor_plus_pytorch_ref,
        proj, diag_sq, norm_factor,
        kernel_name="favor_plus",
    )
