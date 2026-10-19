# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
NASC Attention Autograd Interface

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
        with open(curr_dir / "nasc_attention_binding.cpp", "r", encoding="utf-8") as f: cpp_src = f.read()
        with open(curr_dir / "nasc_attention.cu", "r", encoding="utf-8") as f: cuda_src = f.read()
        _MODULE = load_cuda_module("cuda_nasc", cpp_src, cuda_src, ["nasc_fwd"])
    return _MODULE

def nasc_attention_ref(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, decay: torch.Tensor) -> torch.Tensor:
    B, H, N, D = q.shape
    out = torch.empty_like(q)
    s = torch.zeros(B, H, D, device=q.device, dtype=q.dtype)
    for t in range(N):
        s = decay[:, :, t, :] * s + k[:, :, t, :] * v[:, :, t, :]
        out[:, :, t, :] = q[:, :, t, :] * s
    return out

def nasc_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, decay: torch.Tensor) -> torch.Tensor:
    def cuda_fn(q, k, v, decay):
        mod = _get_module()
        return mod.nasc_fwd(q.contiguous(), k.contiguous(), v.contiguous(), decay.contiguous())
    return dispatch(cuda_fn, nasc_attention_ref, nasc_attention_ref, q, k, v, decay=decay, kernel_name="nasc_attention")
