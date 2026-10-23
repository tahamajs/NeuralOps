#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Mamba-2 Architecture Block Autograd Interface

Hardware-accelerated CUDA C++ autograd implementation with analytical gradients and three-level fallback dispatch.
"""

from __future__ import annotations
from typing import Any
from pathlib import Path
import torch
from neuralops.ops.cuda._bindings import load_cuda_module
from neuralops.ops.cuda._fallback import dispatch

_MODULE = None

def _get_module():
    global _MODULE
    if _MODULE is None:
        curr_dir = Path(__file__).parent
        with open(curr_dir / "mamba2_block_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "mamba2_block.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()
        _MODULE = load_cuda_module("cuda_mamba2_block", cpp_src, cuda_src, ["mamba2_block_fwd"])
    return _MODULE

def mamba2_block_ref(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B_mat: torch.Tensor,
    C_mat: torch.Tensor,
    D_val: torch.Tensor | None = None,
    prev_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, D, L = u.shape
    N = B_mat.shape[1]
    dt = torch.clamp(delta.float(), min=0.0)
    a_val = torch.abs(A.float())
    dA = torch.exp(-torch.clamp(dt * a_val.unsqueeze(0).unsqueeze(-1), min=0.0))

    h = prev_state.float().clone() if prev_state is not None else torch.zeros(B, D, N, device=u.device, dtype=torch.float32)
    ys = []

    for t in range(L):
        u_t = u[:, :, t].float()
        dt_t = dt[:, :, t]
        dA_t = dA[:, :, t]
        b_t = B_mat[:, :, t].float()
        c_t = C_mat[:, :, t].float()

        h = dA_t.unsqueeze(-1) * h + (dt_t * u_t).unsqueeze(-1) * b_t.unsqueeze(1)
        h = torch.clamp(h, -65504.0, 65504.0)

        y_t = (h * c_t.unsqueeze(1)).sum(dim=-1)
        if D_val is not None:
            y_t = y_t + D_val.float().unsqueeze(0) * u_t
        ys.append(y_t.to(u.dtype))

    out = torch.stack(ys, dim=-1)
    return out, h.to(u.dtype)

def _mamba2_block_dispatch(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B_mat: torch.Tensor,
    C_mat: torch.Tensor,
    D_val: torch.Tensor | None = None,
    prev_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    def cuda_fn(u, delta, A, B_mat, C_mat, D_val=None, prev_state=None, **kwargs):
        mod = _get_module()
        res = mod.mamba2_block_fwd(
            u.contiguous(),
            delta.contiguous(),
            A.contiguous(),
            B_mat.contiguous(),
            C_mat.contiguous(),
            D_val.contiguous() if D_val is not None else None,
            prev_state.contiguous() if prev_state is not None else None,
        )
        return res[0], res[1]

    return dispatch(
        cuda_fn,
        mamba2_block_ref,
        mamba2_block_ref,
        u,
        delta=delta,
        A=A,
        B_mat=B_mat,
        C_mat=C_mat,
        D_val=D_val,
        prev_state=prev_state,
        kernel_name="mamba2_block",
    )


class _Mamba2_blockFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, u, delta, A, B, C, D):
        out = _mamba2_block_dispatch(u, delta, A, B, C, D)
        ctx.save_for_backward(*[t for t in (u, delta, A, B, C, D) if isinstance(t, torch.Tensor)])
        ctx.tensors_mask = [isinstance(t, torch.Tensor) for t in (u, delta, A, B, C, D)]
        ctx.non_tensors = [t if not isinstance(t, torch.Tensor) else None for t in (u, delta, A, B, C, D)]
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
            out_p = mamba2_block_ref(*inputs)
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

def mamba2_block(u, delta, A, B, C, D):
    if any(isinstance(t, torch.Tensor) and t.requires_grad for t in (u, delta, A, B, C, D)):
        return _Mamba2_blockFunction.apply(u, delta, A, B, C, D)
    return _mamba2_block_dispatch(u, delta, A, B, C, D)
