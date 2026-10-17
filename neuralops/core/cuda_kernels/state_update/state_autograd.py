"""
Recurrent Linear Attention Step Interface and 3-Level Fallback.
Target: SM 8.9 (NVIDIA Ada Lovelace RTX 4070 Laptop GPU, 8GB VRAM).
"""

from __future__ import annotations

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
    """Pure PyTorch reference implementation."""
    kv_new = kv_state + torch.einsum("bhk,bhv->bhkv", k_t.float(), v_t.float()).to(kv_state.dtype)
    z_new = z_state + k_t
    num = torch.einsum("bhk,bhkv->bhv", q_t.float(), kv_new.float())
    denom = torch.einsum("bhk,bhk->bh", q_t.float(), z_new.float()).unsqueeze(-1) + eps
    out = (num / denom).to(q_t.dtype)
    return out, kv_new, z_new


def _get_triton_impl():
    try:
        from neuralops.core.triton_kernels.state_update import fused_recurrent_linear_step
        return fused_recurrent_linear_step
    except ImportError:
        return None


def recurrent_linear_step(
    q_t: torch.Tensor,
    k_t: torch.Tensor,
    v_t: torch.Tensor,
    kv_state: torch.Tensor,
    z_state: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """3-Level Dispatched Recurrent Linear Step."""
    def cuda_fn(q, k, v, kv, z, eps=1e-6, **kwargs):
        mod = _get_module()
        # Clone states to avoid modifying inputs unexpectedly if called repeatedly
        kv_copy = kv.clone()
        z_copy = z.clone()
        res = mod.recurrent_linear_step_fwd(
            q.contiguous(), k.contiguous(), v.contiguous(), kv_copy, z_copy, float(eps)
        )
        return res[0], res[1], res[2]

    triton_fn = _get_triton_impl()
    if q_t.shape[-1] > 128:
        cuda_fn = None

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
