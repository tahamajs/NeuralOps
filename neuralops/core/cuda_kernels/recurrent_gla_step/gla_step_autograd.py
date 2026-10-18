from __future__ import annotations
from typing import Any, Tuple, Optional
"""
Recurrent GLA Step Interface with Native CUDA Analytical Backward Pass and 3-Level Fallback.
Target: SM 8.9 (NVIDIA Ada Lovelace RTX 4070 Laptop GPU, 8GB VRAM).
"""

import math
from pathlib import Path
import torch

from neuralops.core.cuda_kernels._bindings import load_cuda_module
from neuralops.core.cuda_kernels._fallback import dispatch

_MODULE = None


def _get_module():
    global _MODULE
    if _MODULE is None:
        curr_dir = Path(__file__).parent
        with open(curr_dir / "gla_step_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "recurrent_gla_step.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()

        _MODULE = load_cuda_module(
            name="cuda_recurrent_gla_step",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["recurrent_gla_step_fwd", "recurrent_gla_step_bwd"],
        )
    return _MODULE


def recurrent_gla_step_pytorch_ref(
    q_t: torch.Tensor,
    k_t: torch.Tensor,
    v_t: torch.Tensor,
    s_state: torch.Tensor,
    alpha_t: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pure PyTorch reference implementation."""
    Dk = q_t.shape[-1]
    scale = 1.0 / math.sqrt(Dk)
    a = alpha_t.unsqueeze(-1).unsqueeze(-1) if alpha_t.dim() == 2 else alpha_t.view(q_t.shape[0], q_t.shape[1], 1, 1)
    s_new = a.float() * s_state.float() + torch.einsum("bhk,bhv->bhkv", k_t.float(), v_t.float())
    out = (torch.einsum("bhk,bhkv->bhv", q_t.float(), s_new) * scale).to(q_t.dtype)
    return out, s_new.to(s_state.dtype)


def _get_triton_impl():
    try:
        from neuralops.core.triton_kernels.state_update import fused_recurrent_gla_step
        return fused_recurrent_gla_step
    except (ImportError, AttributeError):
        return None


def _recurrent_gla_step_dispatch(
    q_t: torch.Tensor,
    k_t: torch.Tensor,
    v_t: torch.Tensor,
    s_state: torch.Tensor,
    alpha_t: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """3-Level Dispatched Recurrent GLA Step."""
    def cuda_fn(q, k, v, s, alpha, **kwargs):
        mod = _get_module()
        s_copy = s.clone()
        res = mod.recurrent_gla_step_fwd(
            q.contiguous(), k.contiguous(), v.contiguous(), s_copy, alpha.contiguous()
        )
        return res[0], res[1]

    triton_fn = _get_triton_impl()

    return dispatch(
        cuda_fn,
        triton_fn,
        recurrent_gla_step_pytorch_ref,
        q_t,
        k_t,
        v_t,
        s_state,
        alpha_t,
        kernel_name="recurrent_gla_step",
    )


class _Recurrent_gla_stepFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, q, k, v, state, alpha):
        out, s_next = _recurrent_gla_step_dispatch(q, k, v, state, alpha)
        ctx.save_for_backward(q, k, v, state, alpha)
        return out, s_next

    @staticmethod
    def backward(ctx: Any, *douts):
        q, k, v, state, alpha = ctx.saved_tensors
        dout = douts[0] if len(douts) > 0 and douts[0] is not None else torch.zeros_like(v)
        ds_next = douts[1] if len(douts) > 1 and douts[1] is not None else None

        try:
            mod = _get_module()
            dq, dk, dv, dstate, dalpha = mod.recurrent_gla_step_bwd(
                dout.contiguous(),
                ds_next.contiguous() if ds_next is not None else None,
                q.contiguous(),
                k.contiguous(),
                v.contiguous(),
                state.contiguous(),
                alpha.contiguous(),
            )
            return (
                dq if q.requires_grad else None,
                dk if k.requires_grad else None,
                dv if v.requires_grad else None,
                dstate if state.requires_grad else None,
                dalpha if alpha.requires_grad else None,
            )
        except Exception:
            q_p = q.detach().requires_grad_(q.requires_grad)
            k_p = k.detach().requires_grad_(k.requires_grad)
            v_p = v.detach().requires_grad_(v.requires_grad)
            s_p = state.detach().requires_grad_(state.requires_grad)
            a_p = alpha.detach().requires_grad_(alpha.requires_grad)
            with torch.enable_grad():
                out_p, next_s_p = recurrent_gla_step_pytorch_ref(q_p, k_p, v_p, s_p, a_p)
            inputs = [q_p, k_p, v_p, s_p, a_p]
            diff_inputs = [inp for inp in inputs if inp.requires_grad]
            if not diff_inputs:
                return None, None, None, None, None
            outputs = []
            grad_outputs = []
            if len(douts) > 0 and douts[0] is not None:
                outputs.append(out_p)
                grad_outputs.append(douts[0])
            if ds_next is not None:
                outputs.append(next_s_p)
                grad_outputs.append(ds_next)
            grads = torch.autograd.grad(outputs, diff_inputs, grad_outputs, allow_unused=True)
            grad_map = dict(zip(diff_inputs, grads))
            return tuple(grad_map.get(inp, None) for inp in inputs)


def recurrent_gla_step(q, k, v, state, alpha):
    if any(isinstance(t, torch.Tensor) and t.requires_grad for t in (q, k, v, state, alpha)):
        return _Recurrent_gla_stepFunction.apply(q, k, v, state, alpha)
    return _recurrent_gla_step_dispatch(q, k, v, state, alpha)
