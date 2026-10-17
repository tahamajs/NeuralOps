"""
Gated Linear Attention (GLA) Scan Interface and 3-Level Fallback.
Target: SM 8.9 (NVIDIA Ada Lovelace RTX 4070 Laptop GPU, 8GB VRAM).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import torch

from neuralops.core.cuda_kernels._bindings import load_cuda_module
from neuralops.core.cuda_kernels._fallback import dispatch

_MODULE = None


def _get_module():
    global _MODULE
    if _MODULE is None:
        curr_dir = Path(__file__).parent
        with open(curr_dir / "gla_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "gla_scan.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()

        _MODULE = load_cuda_module(
            name="cuda_gla_scan",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["gla_scan_fwd"],
        )
    return _MODULE


def gla_scan_pytorch_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Pure PyTorch reference implementation."""
    B, H, L, D = q.shape
    scale = 1.0 / math.sqrt(D)
    S = torch.zeros((B, H, D, D), device=q.device, dtype=torch.float32)
    ys = []
    for t in range(L):
        q_t = q[:, :, t, :].float()
        k_t = k[:, :, t, :].float()
        v_t = v[:, :, t, :].float()
        a_t = alpha[:, :, t].unsqueeze(-1).unsqueeze(-1).float() if alpha is not None else 1.0
        kv = torch.matmul(k_t.unsqueeze(-1), v_t.unsqueeze(-2))
        S = a_t * S + kv
        y_t = torch.matmul(q_t.unsqueeze(-2), S).squeeze(-2) * scale
        ys.append(y_t)
    return torch.stack(ys, dim=2).to(q.dtype)


def _get_triton_impl():
    try:
        from neuralops.core.triton_kernels.gla_scan import gla_scan_triton
        return gla_scan_triton
    except (ImportError, AttributeError):
        return None


def gla_scan(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """3-Level Dispatched GLA Scan."""
    def cuda_fn(q, k, v, alpha=None, **kwargs):
        mod = _get_module()
        return mod.gla_scan_fwd(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            alpha.contiguous() if alpha is not None else None,
        )

    triton_fn = _get_triton_impl()
    if q.shape[-1] > 128:
        cuda_fn = None

    return dispatch(
        cuda_fn,
        triton_fn,
        gla_scan_pytorch_ref,
        q,
        k,
        v,
        alpha=alpha,
        kernel_name="gla_scan",
    )
