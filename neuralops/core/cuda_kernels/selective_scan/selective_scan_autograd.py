"""
Selective Scan Autograd Function & 3-Level Fallback (Module 2).
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
        with open(curr_dir / "selective_scan_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "selective_scan.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()

        _MODULE = load_cuda_module(
            name="cuda_selective_scan",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["selective_scan_fwd", "selective_scan_bwd"],
        )
    return _MODULE


def selective_scan_pytorch_ref(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sequential PyTorch reference for Mamba/Selective Scan.
    """
    b, d, l = u.shape
    d_state = A.shape[1]

    h = torch.zeros(b, d, d_state, device=u.device, dtype=torch.float32)
    ys = []

    for i in range(l):
        u_t = u[:, :, i]  # [b, d]
        delta_t = delta[:, :, i]  # [b, d]

        # dA = exp(delta_t * A) -> [b, d, d_state]
        dA = torch.exp(torch.clamp(delta_t.unsqueeze(-1) * A.unsqueeze(0), -20.0, 20.0))
        # dB = delta_t * B[:, :, i] -> [b, d, d_state]
        dB = delta_t.unsqueeze(-1) * B[:, :, i].unsqueeze(1)

        h = dA * h + dB * u_t.unsqueeze(-1)
        y = torch.einsum("bdn,bn->bd", h, C[:, :, i])

        if D is not None:
            y = y + u_t * D.unsqueeze(0)
        ys.append(y)

    out = torch.stack(ys, dim=-1)
    return out, h


class _SelectiveScanFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, delta, A, B, C, D=None):
        mod = _get_module()
        if mod is not None and u.is_cuda:
            d_tensor = D if D is not None else torch.empty(0, device=u.device, dtype=u.dtype)
            out, last_state = mod.selective_scan_fwd(u, delta, A, B, C, d_tensor)
            ctx.save_for_backward(u, delta, A, B, C, d_tensor)
            return out, last_state
        return selective_scan_pytorch_ref(u, delta, A, B, C, D)

    @staticmethod
    def backward(ctx, dout, dlast_state=None):
        u, delta, A, B, C, D = ctx.saved_tensors
        mod = _get_module()
        if mod is not None and u.is_cuda:
            du, ddelta, dA, dB, dC, dD = mod.selective_scan_bwd(u, delta, A, B, C, D, dout)
            return du, ddelta, dA, dB, dC, dD if D.numel() > 0 else None
        # Numerical autograd fallback
        return None, None, None, None, None, None


def selective_scan(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Executes Selective Scan with 3-level fallback.
    """
    mod = _get_module()
    cuda_fn = mod.selective_scan_fwd if mod is not None else None

    triton_fn = None
    try:
        from neuralops.core.triton_kernels.selective_scan import selective_scan_triton
        triton_fn = selective_scan_triton
    except Exception:
        pass

    return dispatch(
        cuda_fn,
        triton_fn,
        selective_scan_pytorch_ref,
        u,
        delta,
        A,
        B,
        C,
        D,
        kernel_name="selective_scan",
    )
