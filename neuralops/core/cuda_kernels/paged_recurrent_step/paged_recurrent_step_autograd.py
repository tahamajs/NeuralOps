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
        with open(curr_dir / "paged_recurrent_step_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "paged_recurrent_step.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()
        _MODULE = load_cuda_module(
            name="cuda_paged_recurrent_step",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["paged_recurrent_step_fwd"],
        )
    return _MODULE

from neuralops.core.triton_kernels.paged_attention import paged_recurrent_state_step_triton

def paged_recurrent_step_pytorch_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    page_pool: torch.Tensor,
    block_tables: torch.Tensor,
    decay: float = 1.0,
    page_slot: int = 0,
) -> torch.Tensor:
    B, H, D = q.shape
    if block_tables.dim() == 1:
        phys_pages = block_tables
    else:
        phys_pages = block_tables[:, page_slot]
    S_curr = page_pool[phys_pages]
    kv = torch.matmul(k.unsqueeze(-1), v.unsqueeze(-2))
    S_next = decay * S_curr + kv
    return torch.matmul(q.unsqueeze(-2), S_next).squeeze(-2)

paged_recurrent_step_ref = paged_recurrent_step_pytorch_ref
paged_recurrent_state_step_pytorch = paged_recurrent_step_pytorch_ref

def _paged_recurrent_step_fwd_dispatch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    page_pool: torch.Tensor,
    block_tables: torch.Tensor,
    decay: float = 1.0,
    page_slot: int = 0,
) -> torch.Tensor:
    def cuda_fn(q, k, v, page_pool, block_tables, decay=1.0, page_slot=0, **kwargs):
        mod = _get_module()
        pp_copy = page_pool.clone()
        return mod.paged_recurrent_step_fwd(
            pp_copy,
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            block_tables.contiguous(),
            int(page_slot),
            float(decay),
        )
    return dispatch(
        cuda_fn,
        paged_recurrent_state_step_triton,
        paged_recurrent_step_pytorch_ref,
        q, k, v, page_pool, block_tables, decay=decay, page_slot=page_slot,
        kernel_name="paged_recurrent_step",
    )

class _PagedRecurrentStepFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, q, k, v, page_pool, block_tables, decay: float = 1.0, page_slot: int = 0):
        out = _paged_recurrent_step_fwd_dispatch(q, k, v, page_pool, block_tables, decay, page_slot)
        ctx.save_for_backward(q, k, v, page_pool, block_tables)
        ctx.decay = float(decay)
        ctx.page_slot = int(page_slot)
        return out

    @staticmethod
    def backward(ctx: Any, *douts):
        q, k, v, page_pool, block_tables = ctx.saved_tensors
        q_p = q.detach().requires_grad_(q.requires_grad)
        k_p = k.detach().requires_grad_(k.requires_grad)
        v_p = v.detach().requires_grad_(v.requires_grad)
        pp_p = page_pool.detach().requires_grad_(page_pool.requires_grad)

        with torch.enable_grad():
            out_p = paged_recurrent_step_pytorch_ref(q_p, k_p, v_p, pp_p, block_tables, ctx.decay, ctx.page_slot)

        grads = torch.autograd.grad(out_p, [q_p, k_p, v_p, pp_p], dout, allow_unused=True)
        return grads[0], grads[1], grads[2], grads[3], None, None, None

def paged_recurrent_step(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    page_pool: torch.Tensor,
    block_tables: torch.Tensor,
    decay: float = 1.0,
    page_slot: int = 0,
) -> torch.Tensor:
    if any(t.requires_grad for t in (q, k, v, page_pool) if isinstance(t, torch.Tensor)):
        return _PagedRecurrentStepFunction.apply(q, k, v, page_pool, block_tables, decay, page_slot)
    return _paged_recurrent_step_fwd_dispatch(q, k, v, page_pool, block_tables, decay=decay, page_slot=page_slot)
