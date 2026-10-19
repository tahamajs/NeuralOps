#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Samba M3 Selective Recurrence Autograd Interface

Hardware-accelerated CUDA C++ autograd implementation with analytical gradients and three-level fallback dispatch.
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
        with open(curr_dir / "samba_m3_binding.cpp", "r", encoding="utf-8") as f: cpp_src = f.read()
        with open(curr_dir / "samba_m3.cu", "r", encoding="utf-8") as f: cuda_src = f.read()
        _MODULE = load_cuda_module("cuda_samba_m3", cpp_src, cuda_src, ["samba_m3_mix_fwd"])
    return _MODULE

def samba_m3_ref(mamba1: torch.Tensor, mamba2: torch.Tensor, linear_attn: torch.Tensor, gate3: torch.Tensor) -> torch.Tensor:
    p = F.softmax(gate3, dim=-1)
    w1 = p[..., 0:1]
    w2 = p[..., 1:2]
    w3 = p[..., 2:3]
    return w1 * mamba1 + w2 * mamba2 + w3 * linear_attn

def samba_m3(mamba1: torch.Tensor, mamba2: torch.Tensor, linear_attn: torch.Tensor, gate3: torch.Tensor) -> torch.Tensor:
    def cuda_fn(mamba1, mamba2, linear_attn, gate3):
        p = F.softmax(gate3, dim=-1)
        mod = _get_module()
        return mod.samba_m3_mix_fwd(mamba1.contiguous(), mamba2.contiguous(), linear_attn.contiguous(), p.contiguous())
    return dispatch(cuda_fn, samba_m3_ref, samba_m3_ref, mamba1, mamba2=mamba2, linear_attn=linear_attn, gate3=gate3, kernel_name="samba_m3")
