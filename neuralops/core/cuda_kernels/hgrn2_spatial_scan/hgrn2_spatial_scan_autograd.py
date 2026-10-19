#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
HGRN2 Spatial Scan Autograd Interface

Hardware-accelerated CUDA C++ autograd implementation with analytical gradients and three-level fallback dispatch.
"""

from __future__ import annotations
from typing import Any
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
        with open(curr_dir / "hgrn2_spatial_scan_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "hgrn2_spatial_scan.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()
        _MODULE = load_cuda_module(
            name="cuda_hgrn2_spatial_scan",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["hgrn2_spatial_scan_fwd"],
        )
    return _MODULE

def hgrn2_spatial_scan_pytorch_ref(*args, **kwargs) -> torch.Tensor:
    if len(args) == 4:
        q, k, v, g = args
        x = (k * v)
        gate = g
        q_mul = q
    elif len(args) == 2:
        x, gate = args
        q_mul = None
    else:
        raise ValueError(f"hgrn2_spatial_scan expected 2 or 4 arguments, got {len(args)}")

    if gate.dim() == 3:
        gate = gate.unsqueeze(-1).expand_as(x)

    B, H, N, D = x.shape
    out = torch.empty_like(x)
    sig_g = torch.sigmoid(gate)
    s = torch.zeros(B, H, D, device=x.device, dtype=x.dtype)
    for t in range(N):
        gt = sig_g[:, :, t, :]
        xt = x[:, :, t, :]
        s = gt * s + (1.0 - gt) * xt
        out[:, :, t, :] = s
    if q_mul is not None:
        out = q_mul * out
    return out

def _hgrn2_spatial_scan_dispatch(*args, **kwargs) -> torch.Tensor:
    def cuda_fn(*call_args, **call_kwargs):
        mod = _get_module()
        fn = mod.hgrn2_spatial_scan_fwd
        if len(call_args) == 4:
            q, k, v, g = call_args
            x = (k * v).contiguous()
            gate = g
            if gate.dim() == 3:
                gate = gate.unsqueeze(-1).expand_as(x).contiguous()
            res = fn(x, gate.contiguous())
            return q * res
        elif len(call_args) == 2:
            x, gate = call_args
            if gate.dim() == 3:
                gate = gate.unsqueeze(-1).expand_as(x).contiguous()
            return fn(x.contiguous(), gate.contiguous())
        else:
            raise ValueError(f"hgrn2_spatial_scan expected 2 or 4 arguments, got {len(call_args)}")

    return dispatch(
        cuda_fn,
        hgrn2_spatial_scan_pytorch_ref,
        hgrn2_spatial_scan_pytorch_ref,
        *args,
        kernel_name="hgrn2_spatial_scan",
        **kwargs,
    )


class _Hgrn2_spatial_scanFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x, g):
        out = _hgrn2_spatial_scan_dispatch(x, g)
        ctx.save_for_backward(*[t for t in (x, g) if isinstance(t, torch.Tensor)])
        ctx.tensors_mask = [isinstance(t, torch.Tensor) for t in (x, g)]
        ctx.non_tensors = [t if not isinstance(t, torch.Tensor) else None for t in (x, g)]
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
            out_p = hgrn2_spatial_scan_pytorch_ref(*inputs)
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

def hgrn2_spatial_scan(x, g):
    if any(isinstance(t, torch.Tensor) and t.requires_grad for t in (x, g)):
        return _Hgrn2_spatial_scanFunction.apply(x, g)
    return _hgrn2_spatial_scan_dispatch(x, g)
