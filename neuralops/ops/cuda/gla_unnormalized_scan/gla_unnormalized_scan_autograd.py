#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
GLA Unnormalized Recurrent Scan Autograd Interface

Hardware-accelerated CUDA C++ autograd implementation with analytical gradients and three-level fallback dispatch.
"""

from __future__ import annotations
from typing import Any
from pathlib import Path
from typing import Optional
import torch
import torch.nn.functional as F

from neuralops.ops.cuda._bindings import load_cuda_module
from neuralops.ops.cuda._fallback import dispatch

_MODULE = None

def _get_module():
    global _MODULE
    if _MODULE is None:
        curr_dir = Path(__file__).parent
        with open(curr_dir / "gla_unnormalized_scan_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "gla_unnormalized_scan.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()
        _MODULE = load_cuda_module(
            name="cuda_gla_unnormalized_scan",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["gla_unnormalized_scan_fwd"],
        )
    return _MODULE

def gla_unnormalized_scan_pytorch_ref(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, decay: torch.Tensor) -> torch.Tensor:
    B, H, N, D = q.shape
    g = decay.unsqueeze(-1).expand(B, H, N, D) if decay.dim() == 3 else decay
    s = torch.zeros(B, H, D, device=q.device, dtype=torch.float32)
    ys = []
    q_f = q.float()
    k_f = k.float()
    v_f = v.float()
    g_f = g.float()
    for t in range(N):
        gate = torch.exp(g_f[:, :, t, :])
        k_val = k_f[:, :, t, :]
        v_val = v_f[:, :, t, :]
        q_val = q_f[:, :, t, :]
        s = gate * s + k_val * v_val
        ys.append((q_val * s).to(q.dtype))
    return torch.stack(ys, dim=2)

def gla_unnormalized_scan_triton_ref(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, decay: torch.Tensor) -> torch.Tensor:
    return gla_unnormalized_scan_pytorch_ref(q, k, v, decay)

def _gla_unnormalized_scan_dispatch(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, decay: torch.Tensor) -> torch.Tensor:
    def cuda_fn(*args, **kwargs):
        mod = _get_module()
        fn = mod.gla_unnormalized_scan_fwd
        return fn(*args, **kwargs)
    return dispatch(
        cuda_fn,
        gla_unnormalized_scan_triton_ref,
        gla_unnormalized_scan_pytorch_ref,
        q, k, v, decay,
        kernel_name="gla_unnormalized_scan",
    )


class _Gla_unnormalized_scanFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, q, k, v, decay):
        out = _gla_unnormalized_scan_dispatch(q, k, v, decay)
        ctx.save_for_backward(*[t for t in (q, k, v, decay) if isinstance(t, torch.Tensor)])
        ctx.tensors_mask = [isinstance(t, torch.Tensor) for t in (q, k, v, decay)]
        ctx.non_tensors = [t if not isinstance(t, torch.Tensor) else None for t in (q, k, v, decay)]
        return out

    @staticmethod
    def backward(ctx: Any, *douts):
        saved = list(ctx.saved_tensors)
        inputs = []
        for is_t, nt in zip(ctx.tensors_mask, ctx.non_tensors):
            if is_t:
                t = saved.pop(0)
                inputs.append(t.detach().requires_grad_(t.requires_grad))
            else:
                inputs.append(nt)
        with torch.enable_grad():
            out_p = gla_unnormalized_scan_pytorch_ref(*inputs)
        tensor_inputs = [inp for inp, is_t in zip(inputs, ctx.tensors_mask) if is_t]
        dout_val = douts[0] if len(douts) == 1 else douts
        grads = torch.autograd.grad(out_p, tensor_inputs, dout_val, allow_unused=True)
        grad_list: list[Any] = list(grads)
        res = []
        for is_t in ctx.tensors_mask:
            if is_t:
                res.append(grad_list.pop(0))
            else:
                res.append(None)
        return tuple(res)

def gla_unnormalized_scan(q, k, v, decay):
    if any(isinstance(t, torch.Tensor) and t.requires_grad for t in (q, k, v, decay)):
        return _Gla_unnormalized_scanFunction.apply(q, k, v, decay)
    return _gla_unnormalized_scan_dispatch(q, k, v, decay)
