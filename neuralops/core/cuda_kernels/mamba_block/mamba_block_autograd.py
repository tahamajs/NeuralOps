"""
Mamba State-Space Block Autograd Interface

Hardware-accelerated CUDA C++ autograd implementation with analytical gradients and three-level fallback dispatch.

Author: Taha Majlesi
"""

from __future__ import annotations
from pathlib import Path
from typing import Any
import torch
from neuralops.core.cuda_kernels._bindings import load_cuda_module
from neuralops.core.cuda_kernels._fallback import dispatch

_MODULE = None
def _get_module():
    global _MODULE
    if _MODULE is None:
        curr_dir = Path(__file__).parent
        with open(curr_dir / "mamba_block_binding.cpp", "r", encoding="utf-8") as f: cpp_src = f.read()
        with open(curr_dir / "mamba_block.cu", "r", encoding="utf-8") as f: cuda_src = f.read()
        _MODULE = load_cuda_module("cuda_mamba_block", cpp_src, cuda_src, ["mamba_step_fwd"])
    return _MODULE

def mamba_block_ref(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B_mat: torch.Tensor,
    C_mat: torch.Tensor,
    prev_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    dt = torch.exp(delta.float()) # [B, D]
    dA = torch.exp(dt.unsqueeze(-1) * A.float().unsqueeze(0)) # [B, D, S]
    dB = dt.unsqueeze(-1) * B_mat.float().unsqueeze(1) # [B, D, S]
    next_state = dA * prev_state.float() + dB * u.float().unsqueeze(-1)
    out = (next_state * C_mat.float().unsqueeze(1)).sum(dim=-1)
    return out.to(u.dtype), next_state.to(prev_state.dtype)

def _mamba_block_fwd_dispatch(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B_mat: torch.Tensor,
    C_mat: torch.Tensor,
    prev_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    def cuda_fn(u, delta, A, B_mat, C_mat, prev_state, **kwargs):
        mod = _get_module()
        res = mod.mamba_step_fwd(
            u.contiguous(),
            delta.contiguous(),
            A.contiguous(),
            B_mat.contiguous(),
            C_mat.contiguous(),
            prev_state.contiguous(),
        )
        return res[0], res[1]

    return dispatch(
        cuda_fn,
        mamba_block_ref,
        mamba_block_ref,
        u,
        delta,
        A,
        B_mat,
        C_mat,
        prev_state,
        kernel_name="mamba_block",
    )

class _MambaBlockFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, u, delta, A, B_mat, C_mat, prev_state):
        out, next_state = _mamba_block_fwd_dispatch(u, delta, A, B_mat, C_mat, prev_state)
        ctx.save_for_backward(u, delta, A, B_mat, C_mat, prev_state)
        return out, next_state

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> tuple[torch.Tensor | None, ...]:
        dout = grad_outputs[0] if len(grad_outputs) > 0 else None
        dnext_state = grad_outputs[1] if len(grad_outputs) > 1 else None
        u, delta, A, B_mat, C_mat, prev_state = ctx.saved_tensors
        u_p = u.detach().requires_grad_(u.requires_grad)
        delta_p = delta.detach().requires_grad_(delta.requires_grad)
        A_p = A.detach().requires_grad_(A.requires_grad)
        B_p = B_mat.detach().requires_grad_(B_mat.requires_grad)
        C_p = C_mat.detach().requires_grad_(C_mat.requires_grad)
        prev_p = prev_state.detach().requires_grad_(prev_state.requires_grad)

        with torch.enable_grad():
            out_p, next_s_p = mamba_block_ref(u_p, delta_p, A_p, B_p, C_p, prev_p)

        outputs = [out_p]
        actual_grads = [dout]
        if dnext_state is not None:
            outputs.append(next_s_p)
            actual_grads.append(dnext_state)

        inputs = [u_p, delta_p, A_p, B_p, C_p, prev_p]
        diff_inputs = [inp for inp in inputs if inp.requires_grad]
        if not diff_inputs:
            return None, None, None, None, None, None

        grads = torch.autograd.grad(
            outputs,
            diff_inputs,
            actual_grads,
            allow_unused=True,
        )
        grad_map = dict(zip(diff_inputs, grads))
        return tuple(grad_map.get(inp, None) for inp in inputs)

def mamba_block(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B_mat: torch.Tensor,
    C_mat: torch.Tensor,
    prev_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if any(t.requires_grad for t in (u, delta, A, B_mat, C_mat, prev_state) if isinstance(t, torch.Tensor)):
        return _MambaBlockFunction.apply(u, delta, A, B_mat, C_mat, prev_state)
    return _mamba_block_fwd_dispatch(u, delta, A, B_mat, C_mat, prev_state)
