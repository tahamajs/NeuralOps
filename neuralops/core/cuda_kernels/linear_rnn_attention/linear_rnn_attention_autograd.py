#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Linear RNN Attention Autograd Interface

Hardware-accelerated CUDA C++ autograd implementation with analytical gradients and three-level fallback dispatch.
"""

from __future__ import annotations
from typing import Any
from pathlib import Path
import torch
from neuralops.core.cuda_kernels._bindings import load_cuda_module
from neuralops.core.cuda_kernels._fallback import dispatch

_MODULE = None
def _get_module():
    global _MODULE
    if _MODULE is None:
        curr_dir = Path(__file__).parent
        with open(curr_dir / "linear_rnn_attention_binding.cpp", "r", encoding="utf-8") as f: cpp_src = f.read()
        with open(curr_dir / "linear_rnn_attention.cu", "r", encoding="utf-8") as f: cuda_src = f.read()
        _MODULE = load_cuda_module("cuda_linear_rnn_attention", cpp_src, cuda_src, ["linear_rnn_fwd"])
    return _MODULE

def linear_rnn_attention_ref(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    B, N, D = x.shape
    out = torch.empty_like(x)
    h = torch.zeros(B, D, device=x.device, dtype=x.dtype)
    for t in range(N):
        h = a[:, t, :] * h + b[:, t, :] * x[:, t, :]
        out[:, t, :] = h
    return out

def _linear_rnn_attention_dispatch(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    def cuda_fn(x, a, b):
        mod = _get_module()
        return mod.linear_rnn_fwd(x.contiguous(), a.contiguous(), b.contiguous())
    return dispatch(cuda_fn, linear_rnn_attention_ref, linear_rnn_attention_ref, x, a=a, b=b, kernel_name="linear_rnn_attention")


class _Linear_rnn_attentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x, a, b):
        out = _linear_rnn_attention_dispatch(x, a, b)
        ctx.save_for_backward(*[t for t in (x, a, b) if isinstance(t, torch.Tensor)])
        ctx.tensors_mask = [isinstance(t, torch.Tensor) for t in (x, a, b)]
        ctx.non_tensors = [t if not isinstance(t, torch.Tensor) else None for t in (x, a, b)]
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
            out_p = linear_rnn_attention_ref(*inputs)
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

def linear_rnn_attention(x, a, b):
    if any(isinstance(t, torch.Tensor) and t.requires_grad for t in (x, a, b)):
        return _Linear_rnn_attentionFunction.apply(x, a, b)
    return _linear_rnn_attention_dispatch(x, a, b)
