# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Recurrent HGRN2 Step Autograd Interface

Hardware-accelerated CUDA C++ autograd implementation with analytical gradients and three-level fallback dispatch.
"""

from __future__ import annotations
from typing import Any
"""
Recurrent HGRN2 Step Interface and 3-Level Fallback.
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
        with open(curr_dir / "hgrn2_step_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "recurrent_hgrn2_step.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()

        _MODULE = load_cuda_module(
            name="cuda_recurrent_hgrn2_step",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["recurrent_hgrn2_step_fwd"],
        )
    return _MODULE


def recurrent_hgrn2_step_pytorch_ref(
    x_t: torch.Tensor,
    g_t: torch.Tensor,
    alpha_t: torch.Tensor,
    h_prev: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pure PyTorch reference implementation."""
    h_new = alpha_t * h_prev + (1.0 - alpha_t) * x_t
    y_t = h_new * g_t
    return y_t, h_new


def _get_triton_impl():
    try:
        from neuralops.core.triton_kernels.state_update import fused_recurrent_hgrn2_step
        return fused_recurrent_hgrn2_step
    except (ImportError, AttributeError):
        return None


def _recurrent_hgrn2_step_dispatch(
    x_t: torch.Tensor,
    g_t: torch.Tensor,
    alpha_t: torch.Tensor,
    h_prev: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """3-Level Dispatched Recurrent HGRN2 Step."""
    def cuda_fn(x, g, a, h, **kwargs):
        mod = _get_module()
        h_copy = h.clone()
        res = mod.recurrent_hgrn2_step_fwd(
            x.contiguous(), g.contiguous(), a.contiguous(), h_copy
        )
        return res[0], res[1]

    triton_fn = _get_triton_impl()

    return dispatch(
        cuda_fn,
        triton_fn,
        recurrent_hgrn2_step_pytorch_ref,
        x_t,
        g_t,
        alpha_t,
        h_prev,
        kernel_name="recurrent_hgrn2_step",
    )


class _Recurrent_hgrn2_stepFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x, g, alpha, prev_state):
        out = _recurrent_hgrn2_step_dispatch(x, g, alpha, prev_state)
        ctx.save_for_backward(*[t for t in (x, g, alpha, prev_state) if isinstance(t, torch.Tensor)])
        ctx.tensors_mask = [isinstance(t, torch.Tensor) for t in (x, g, alpha, prev_state)]
        ctx.non_tensors = [t if not isinstance(t, torch.Tensor) else None for t in (x, g, alpha, prev_state)]
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
            out_p = recurrent_hgrn2_step_pytorch_ref(*inputs)
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

def recurrent_hgrn2_step(x, g, alpha, prev_state):
    if any(isinstance(t, torch.Tensor) and t.requires_grad for t in (x, g, alpha, prev_state)):
        return _Recurrent_hgrn2_stepFunction.apply(x, g, alpha, prev_state)
    return _recurrent_hgrn2_step_dispatch(x, g, alpha, prev_state)
