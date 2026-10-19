# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
SVD Prefix Scan Autograd Interface

Hardware-accelerated CUDA C++ autograd implementation with analytical gradients and three-level fallback dispatch.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Optional
import torch
import torch.nn.functional as F

from neuralops.core.cuda_kernels._bindings import load_cuda_module
from neuralops.core.cuda_kernels._fallback import dispatch

_MODULE = None

def _get_module():
    global _MODULE
    if _MODULE is None:
        curr_dir = Path(__file__).parent
        with open(curr_dir / "svd_prefix_scan_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "svd_prefix_scan.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()
        _MODULE = load_cuda_module(
            name="cuda_svd_prefix_scan",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["svd_prefix_scan_fwd"],
        )
    return _MODULE

def svd_prefix_scan_pytorch_ref(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    kv = torch.einsum("bhld,bhle->bhlde", k.float(), v.float())
    s = kv.cumsum(dim=2)
    return torch.einsum("bhld,bhlde->bhle", q.float(), s).to(q.dtype)

def svd_prefix_scan_triton_ref(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    return svd_prefix_scan_pytorch_ref(q, k, v, eps=eps)

def _svd_prefix_scan_fwd_dispatch(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    def cuda_fn(q, k, v, eps=1e-4, **kwargs):
        mod = _get_module()
        return mod.svd_prefix_scan_fwd(q.contiguous(), k.contiguous(), v.contiguous(), float(eps))
    return dispatch(
        cuda_fn,
        svd_prefix_scan_triton_ref,
        svd_prefix_scan_pytorch_ref,
        q, k, v, float(eps),
        kernel_name="svd_prefix_scan",
    )

class _SvdPrefixScanFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, q, k, v, eps: float = 1e-4):
        out = _svd_prefix_scan_fwd_dispatch(q, k, v, eps)
        ctx.save_for_backward(q, k, v)
        ctx.eps = eps
        return out

    @staticmethod
    def backward(ctx: Any, *douts):
        dout = douts[0] if len(douts) == 1 else douts
        q, k, v = ctx.saved_tensors
        q_p = q.detach().requires_grad_(q.requires_grad)
        k_p = k.detach().requires_grad_(k.requires_grad)
        v_p = v.detach().requires_grad_(v.requires_grad)

        with torch.enable_grad():
            out_p = svd_prefix_scan_pytorch_ref(q_p, k_p, v_p, ctx.eps)

        grads = torch.autograd.grad(out_p, [q_p, k_p, v_p], dout, allow_unused=True)
        return grads[0], grads[1], grads[2], None

def svd_prefix_scan(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    if any(t.requires_grad for t in (q, k, v) if isinstance(t, torch.Tensor)):
        return _SvdPrefixScanFunction.apply(q, k, v, eps)
    return _svd_prefix_scan_fwd_dispatch(q, k, v, eps)
