#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Temporal Pattern Attention Autograd Interface

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
        with open(curr_dir / "tpa_attention_binding.cpp", "r", encoding="utf-8") as f: cpp_src = f.read()
        with open(curr_dir / "tpa_attention.cu", "r", encoding="utf-8") as f: cuda_src = f.read()
        _MODULE = load_cuda_module("cuda_tpa", cpp_src, cuda_src, ["tpa_fwd"])
    return _MODULE

def tpa_attention_ref(h_states: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return torch.bmm(weights.unsqueeze(1), h_states).squeeze(1)

def tpa_attention(h_states: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    def cuda_fn(h_states, weights):
        mod = _get_module()
        return mod.tpa_fwd(h_states.contiguous(), weights.contiguous())
    return dispatch(cuda_fn, tpa_attention_ref, tpa_attention_ref, h_states, weights=weights, kernel_name="tpa_attention")
