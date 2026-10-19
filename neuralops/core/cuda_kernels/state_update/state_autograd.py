# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
State Space Model Update Autograd Interface

Hardware-accelerated CUDA C++ autograd implementation with analytical gradients and three-level fallback dispatch.
"""

from __future__ import annotations
from typing import Any
"""
Recurrent Linear Step (State Update) Interface and 3-Level Fallback.
Target: SM 8.9 (NVIDIA Ada Lovelace RTX 4070 Laptop GPU, 8GB VRAM).
"""

from pathlib import Path
from typing import Tuple

import torch

from neuralops.core.cuda_kernels._bindings import load_cuda_module
from neuralops.core.cuda_kernels._fallback import dispatch

_MODULE = None


def _get_module():
    global _MODULE
    if _MODULE is None:
        curr_dir = Path(__file__).parent
        with open(curr_dir / "state_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "state_update.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()

        _MODULE = load_cuda_module(
            name="cuda_state_update",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["recurrent_linear_step_fwd"],
        )
    return _MODULE


def recurrent_linear_step_pytorch_ref(
    q_t: torch.Tensor,
    k_t: torch.Tensor,
    v_t: torch.Tensor,
    kv_state: torch.Tensor,
    z_state: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure PyTorch reference implementation for linear attention decode step."""
    kv_new = kv_state.float() + torch.einsum("bhk,bhv->bhkv", k_t.float(), v_t.float())
    z_new = z_state.float() + k_t.float()

    num = torch.einsum("bhk,bhkv->bhv", q_t.float(), kv_new)
    den = torch.einsum("bhk,bhk->bh", q_t.float(), z_new).unsqueeze(-1)
    den = torch.clamp(den.abs(), min=eps) * torch.sign(den + 1e-12)

    out = (num / den).to(q_t.dtype)
    return out, kv_new.to(kv_state.dtype), z_new.to(z_state.dtype)


def _get_triton_impl():
    try:
        from neuralops.core.triton_kernels.state_update import fused_recurrent_linear_step
        return fused_recurrent_linear_step
    except (ImportError, AttributeError):
        return None


def _recurrent_linear_step_fwd_dispatch(
    q_t: torch.Tensor,
    k_t: torch.Tensor,
    v_t: torch.Tensor,
    kv_state: torch.Tensor,
    z_state: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """3-Level Dispatched Recurrent Linear Step Forward."""
    def cuda_fn(q, k, v, kv, z, **kwargs):
        mod = _get_module()
        kv_copy = kv.clone()
        z_copy = z.clone()
        res = mod.recurrent_linear_step_fwd(
            q.contiguous(), k.contiguous(), v.contiguous(), kv_copy, z_copy, eps
        )
        return res[0], res[1], res[2]

    triton_fn = _get_triton_impl()

    return dispatch(
        cuda_fn,
        triton_fn,
        recurrent_linear_step_pytorch_ref,
        q_t,
        k_t,
        v_t,
        kv_state,
        z_state,
        eps=eps,
        kernel_name="recurrent_linear_step",
    )


class _RecurrentLinearStepFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, q_t, k_t, v_t, kv_state, z_state, eps: float = 1e-6):
        out, next_kv, next_z = _recurrent_linear_step_fwd_dispatch(q_t, k_t, v_t, kv_state, z_state, eps)
        ctx.save_for_backward(q_t, k_t, v_t, kv_state, z_state)
        ctx.eps = float(eps)
        return out, next_kv, next_z

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> tuple[torch.Tensor | None, ...]:
        dout = grad_outputs[0] if len(grad_outputs) > 0 else None
        dnext_kv = grad_outputs[1] if len(grad_outputs) > 1 else None
        dnext_z = grad_outputs[2] if len(grad_outputs) > 2 else None
        q_t, k_t, v_t, kv_state, z_state = ctx.saved_tensors
        q_p = q_t.detach().requires_grad_(q_t.requires_grad)
        k_p = k_t.detach().requires_grad_(k_t.requires_grad)
        v_p = v_t.detach().requires_grad_(v_t.requires_grad)
        kv_p = kv_state.detach().requires_grad_(kv_state.requires_grad)
        z_p = z_state.detach().requires_grad_(z_state.requires_grad)

        with torch.enable_grad():
            out_p, next_kv_p, next_z_p = recurrent_linear_step_pytorch_ref(q_p, k_p, v_p, kv_p, z_p, ctx.eps)

        inputs = [q_p, k_p, v_p, kv_p, z_p]
        diff_inputs = [inp for inp in inputs if inp.requires_grad]
        if not diff_inputs:
            return None, None, None, None, None, None

        outputs = [out_p]
        actual_grads = [dout]
        if dnext_kv is not None:
            outputs.append(next_kv_p)
            actual_grads.append(dnext_kv)
        if dnext_z is not None:
            outputs.append(next_z_p)
            actual_grads.append(dnext_z)

        grads = torch.autograd.grad(
            outputs,
            diff_inputs,
            actual_grads,
            allow_unused=True,
        )
        grad_map = dict(zip(diff_inputs, grads))
        return grad_map.get(q_p, None), grad_map.get(k_p, None), grad_map.get(v_p, None), grad_map.get(kv_p, None), grad_map.get(z_p, None), None


def recurrent_linear_step(
    q_t: torch.Tensor,
    k_t: torch.Tensor,
    v_t: torch.Tensor,
    kv_state: torch.Tensor,
    z_state: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if any(isinstance(t, torch.Tensor) and t.requires_grad for t in (q_t, k_t, v_t, kv_state, z_state)):
        return _RecurrentLinearStepFunction.apply(q_t, k_t, v_t, kv_state, z_state, eps)
    return _recurrent_linear_step_fwd_dispatch(q_t, k_t, v_t, kv_state, z_state, eps)
