"""
Recurrent GLA Step Interface and 3-Level Fallback.
Target: SM 8.9 (NVIDIA Ada Lovelace RTX 4070 Laptop GPU, 8GB VRAM).
"""

from __future__ import annotations

import math
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
        with open(curr_dir / "gla_step_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "recurrent_gla_step.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()

        _MODULE = load_cuda_module(
            name="cuda_recurrent_gla_step",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["recurrent_gla_step_fwd"],
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


def recurrent_gla_step(
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
