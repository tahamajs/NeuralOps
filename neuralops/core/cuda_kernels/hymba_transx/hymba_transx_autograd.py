# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Hymba TransX Fused Operator Autograd Interface

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
        with open(curr_dir / "hymba_transx_binding.cpp", "r", encoding="utf-8") as f: cpp_src = f.read()
        with open(curr_dir / "hymba_transx.cu", "r", encoding="utf-8") as f: cuda_src = f.read()
        _MODULE = load_cuda_module("cuda_hymba_transx", cpp_src, cuda_src, ["hymba_transx_fwd"])
    return _MODULE

def hymba_transx_ref(x: torch.Tensor, meta_tokens: torch.Tensor, cross_gate: torch.Tensor) -> torch.Tensor:
    meta_mean = meta_tokens.mean(dim=1, keepdim=True)
    g = torch.sigmoid(cross_gate)
    return x + g * meta_mean

def hymba_transx(x: torch.Tensor, meta_tokens: torch.Tensor, cross_gate: torch.Tensor) -> torch.Tensor:
    def cuda_fn(x, meta_tokens, cross_gate):
        mod = _get_module()
        return mod.hymba_transx_fwd(x.contiguous(), meta_tokens.contiguous(), cross_gate.contiguous())
    return dispatch(cuda_fn, hymba_transx_ref, hymba_transx_ref, x, meta_tokens=meta_tokens, cross_gate=cross_gate, kernel_name="hymba_transx")
