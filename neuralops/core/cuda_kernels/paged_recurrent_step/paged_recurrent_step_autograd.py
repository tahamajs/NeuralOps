from __future__ import annotations
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
    """Pure PyTorch reference implementation for paged recurrent step."""
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

def paged_recurrent_step(
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
        # Clone page_pool copy to avoid benchmark in-place state mutation drift
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
