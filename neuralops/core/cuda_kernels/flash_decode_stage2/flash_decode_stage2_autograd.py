#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Flash-Decoding Stage 2 Autograd Interface

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
        with open(curr_dir / "flash_decode_stage2_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "flash_decode_stage2.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()
        _MODULE = load_cuda_module(
            name="cuda_flash_decode_stage2",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["flash_decode_stage2_fwd"],
        )
    return _MODULE

def flash_decode_stage2_pytorch_ref(mid_out, mid_lse):
    glse = torch.logsumexp(mid_lse, dim=-1, keepdim=True)
    alpha = torch.exp(mid_lse - glse).unsqueeze(-1)
    return (alpha * mid_out).sum(dim=2, keepdim=True).to(mid_out.dtype)
def flash_decode_stage2_triton_ref(mid_out, mid_lse):
    return flash_decode_stage2_pytorch_ref(mid_out, mid_lse)

def flash_decode_stage2(mid_out: torch.Tensor, mid_lse: torch.Tensor) -> torch.Tensor:
    def cuda_fn(*args, **kwargs):
        mod = _get_module()
        fn = mod.flash_decode_stage2_fwd
        return fn(*args, **kwargs)
    return dispatch(
        cuda_fn,
        flash_decode_stage2_triton_ref,
        flash_decode_stage2_pytorch_ref,
        mid_out, mid_lse,
        kernel_name="flash_decode_stage2",
    )
