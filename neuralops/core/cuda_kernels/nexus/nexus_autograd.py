# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Nexus Recurrent Operator Autograd Interface

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
        with open(curr_dir / "nexus_binding.cpp", "r", encoding="utf-8") as f: cpp_src = f.read()
        with open(curr_dir / "nexus.cu", "r", encoding="utf-8") as f: cuda_src = f.read()
        _MODULE = load_cuda_module("cuda_nexus", cpp_src, cuda_src, ["nexus_fwd"])
    return _MODULE

def nexus_ref(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    phi_q = torch.tanh(q * tau)
    phi_k = torch.tanh(k * tau)
    B, H, N, D = q.shape
    out = torch.empty_like(q)
    s = torch.zeros(B, H, D, device=q.device, dtype=q.dtype)
    for t in range(N):
        s = s + phi_k[:, :, t, :] * v[:, :, t, :]
        out[:, :, t, :] = phi_q[:, :, t, :] * s
    return out

def nexus(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    def cuda_fn(q, k, v, tau=1.0):
        mod = _get_module()
        return mod.nexus_fwd(q.contiguous(), k.contiguous(), v.contiguous(), float(tau))
    return dispatch(cuda_fn, nexus_ref, nexus_ref, q, k=k, v=v, tau=tau, kernel_name="nexus")
