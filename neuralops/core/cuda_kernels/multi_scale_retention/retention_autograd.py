"""
Multi-Scale Retention Interface and 3-Level Fallback.
Target: SM 8.9 (NVIDIA Ada Lovelace RTX 4070 Laptop GPU, 8GB VRAM).
"""

from __future__ import annotations

from pathlib import Path

import torch

from neuralops.core.cuda_kernels._bindings import load_cuda_module
from neuralops.core.cuda_kernels._fallback import dispatch

_MODULE = None


def _get_module():
    global _MODULE
    if _MODULE is None:
        curr_dir = Path(__file__).parent
        with open(curr_dir / "retention_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "multi_scale_retention.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()

        _MODULE = load_cuda_module(
            name="cuda_multi_scale_retention",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["multi_scale_retention_fwd"],
        )
    return _MODULE


def multi_scale_retention_pytorch_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gammas: torch.Tensor,
) -> torch.Tensor:
    """Pure PyTorch reference implementation."""
    B, H, L, D = q.shape
    S = torch.zeros((B, H, D, D), device=q.device, dtype=torch.float32)
    ys = []
    g = gammas.view(1, H, 1, 1).float()

    for t in range(L):
        q_t = q[:, :, t, :].float()
        k_t = k[:, :, t, :].float()
        v_t = v[:, :, t, :].float()
        kv = torch.matmul(k_t.unsqueeze(-1), v_t.unsqueeze(-2))
        S = g * S + kv
        y_t = torch.matmul(q_t.unsqueeze(-2), S).squeeze(-2)
        ys.append(y_t)

    return torch.stack(ys, dim=2).to(q.dtype)


def _get_triton_impl():
    try:
        from neuralops.core.triton_kernels.linear_attn import multi_scale_retention_triton
        return multi_scale_retention_triton
    except (ImportError, AttributeError):
        return None


def multi_scale_retention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gammas: torch.Tensor,
) -> torch.Tensor:
    """3-Level Dispatched Multi-Scale Retention."""
    def cuda_fn(q, k, v, gammas, **kwargs):
        mod = _get_module()
        return mod.multi_scale_retention_fwd(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            gammas.contiguous(),
        )

    triton_fn = _get_triton_impl()
    if q.shape[-1] > 128:
        cuda_fn = None

    return dispatch(
        cuda_fn,
        triton_fn,
        multi_scale_retention_pytorch_ref,
        q,
        k,
        v,
        gammas,
        kernel_name="multi_scale_retention",
    )
